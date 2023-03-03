#!/usr/bin/python

# Spark-based PageRank calculation using DataFrames.

# Program proceeds as follows:
# We create an dataframe of schema ["paper", "score"]
# and another of the schema ["node", "cited_list"]
# There will be two dataframes of the first type in the progam,
# which will allow us to calculate max score differences
# between consecutive runs. 

# After score calculations, normalized scores and score classes are added

# ---------- Imports ------------- #
# NOTE: Use this for python 2
from __future__ import print_function
import sys

# Pyspark-specific imports
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
# Functions to effectively handle DataFrame data
import pyspark.sql.functions as F
from pyspark.sql import Window

# Diagnostics
import time
# -------------------------------- #

# ----------- Inits -------------- #

# Print output message if not run correctly
###########################################
if len(sys.argv) < 5:
	print ("Usage: pagerank_spark_df.py <input_file> <alpha> <convergence_error> <checkpoint_dir> <(optional)num_partitions> <(optional)checkpointing_mode>")
	sys.exit(0)
###########################################

############################################
# Read input parameters
input_file = sys.argv[1]
alpha      = float(sys.argv[2])
max_error  = float(sys.argv[3])
checkpoint_dir = sys.argv[4]
# Set default partitions to 1
try:
        num_partitions = int(sys.argv[5])
except:
        num_partitions = 1
# Set checkpoint mode / default checkpoint mode
try:
        checkpoint_mode = sys.argv[6]
except:
        checkpoint_mode = 'dfs'		
		
# Set the mode by default as local. 
# If data is read from hdfs we switch to cluster
mode = 'local'
if input_file.startswith('hdfs://'):
	mode = 'distributed'
	
# Set the mode by default as dfs. 
if checkpoint_mode not in ['dfs', 'local']:
	checkpoint_mode = 'dfs'
############################################
# Measure initialization time
initialisation_time = time.time()

# Initialize spark context and spark-specific params
#####################################################################################################
if  mode == 'local':
	conf = SparkConf().setMaster('local').setAppName('Spark PageRank')
else:
	conf = SparkConf().setAppName('Spark PageRank')
# Pass configuration to spark context
sc = SparkContext(conf = conf)
# Session is needed for spark.sql library, in order to use DataFrames
if mode == 'local':
	spark = SparkSession.builder.master('local').appName('Spark PageRank').getOrCreate()
else:
	spark = SparkSession.builder.config('spark.cleaner.referenceTracking.cleanCheckpoints', 'true').appName('Spark PageRank').getOrCreate()

# Turn off logging (possible values: OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL
sc.setLogLevel('OFF')

# Set checkpoint dir
sc.setCheckpointDir(checkpoint_dir)
#####################################################################################################
# Define schema for reading the input file
graph_file_schema = StructType([
	StructField("paper", StringType(), False),
	StructField("citation_data", StringType(), False),
	StructField("prev_score", FloatType(), False),
	StructField("pub_year", IntegerType(), False)		
	])
#####################################################################################################

#####################################################################################################
# Get hdfs file handler object. 
# This solution, although not elegant, does not rely on third party libaries
# Source: https://diogoalexandrefranco.github.io/interacting-with-hdfs-from-pyspark/
URI = sc._gateway.jvm.java.net.URI
Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
# Get master prefix from input file path
master_prefix = '/'.join(input_file.split('/')[:5])
fs = FileSystem.get(URI(master_prefix), sc._jsc.hadoopConfiguration())
#################################
# Set PAGERANK-Specific params
#------------------------------ #
# Set initial error to high value
error 	   = 1000
# Read the initial input
if mode == 'local':
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file).cache()
elif mode == 'distributed':
	print ("Reading input from HDFS...")
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file).repartition(num_partitions, 'paper').cache()
	
	
# Get number of nodes (one node-record per line)
num_nodes = float(input_data.count())
# Random jump probability/initial score
random_jump_prob = float(1)/float(num_nodes)
# Count iterations
iterations = 1
##################################
# Display PR parameterisation
print ("\n# ------------------------------------ #")
print ("Input: ", input_file)
print ("Alpha: ", alpha)
print ("Convergence Error: ", max_error)
print ("Number of nodes: ", num_nodes)
print ("Number of partitions: ", num_partitions) 
print ("Checkpoint mode: " + str(checkpoint_mode))
print ("Checkpoint dir: " + checkpoint_dir)
print ("# ------------------------------------ #\n")

print ("Initialising", end = '')
sys.stdout.flush()
##################################################
# Initialise SPARK Data

# Get a DataFrame with pairs of <node, score>.
# Recall our input format to understand what the following map function does:
# <paper> <tab> <cited_papers|num_cited_papers|score> <tab> <previous_score> <tab> <publication_year>
scores = input_data.select('paper', F.split('citation_data', "\|").alias('citation_data') )\
		   .select('paper', F.expr('element_at(citation_data, size(citation_data))').alias('score')).cache()
		   # .select('paper', F.element_at(F.col('citation_data'), F.expr('size(citation_data)') ).alias('score') ).cache()

# Duplicate scores to keep track of scores in consecutive iterations
previous_scores	= scores.select('paper',F.col('score').alias('previous_score'))

# Continue initialisation message
print(".", end = '')
sys.stdout.flush()

# Get a dataframe of doi - cited list
outlinks = input_data.select('paper', F.split('citation_data', "\|").alias('cited_papers'), 'pub_year')\
		     .select('paper', 'cited_papers', F.expr('size(cited_papers)-2').alias('cited_paper_size'), 'pub_year')\
		     .select('paper', F.expr('slice(cited_papers, 1, cited_paper_size)').alias('cited_papers'), 'pub_year')\
		     .select('paper', F.array_join('cited_papers', '|').alias('cited_papers'), 'pub_year')\
		     .select('paper', F.split('cited_papers', ',').alias('cited_papers'), 'pub_year').repartition(num_partitions, 'paper').cache()

# Create a DataFrame with nodes filtered based on whether they cite others or not
outlinks_actual = outlinks.filter(outlinks['cited_papers'][0] != '0').repartition(num_partitions, 'paper').cache()

# Continue intialisation message
print(".", end = '')
sys.stdout.flush()

# Collect the dangling nodes from the data - cache it since it will be reused
dangling_nodes = outlinks.filter(outlinks.cited_papers[0] == '0').select('paper').repartition(num_partitions, 'paper').cache()

# Continue intialisation message
print(".", end = '')
sys.stdout.flush()

# Continue intialisation message
print("Took: %s seconds!" % (time.time()-initialisation_time))
print("Starting!")
##################################################


# -------------------------------- #

# ------------- Main ------------- #
# Loop until error is less than the threshold given
while error >= max_error:

	# Time the iteration
	iteration_time = time.time()

	# Notify user
	print ("\n# ------------------------------------ #\n")

	# -------------------------------- #
	# 1. Calculate PageRank scores transferred by Dangling Nodes 
	# Join by key with scores DataFram and then sum values
	# This returns a single row with a single column, therefore we specifically ask for [0][0] offset
	dangling_sum = dangling_nodes.join(scores, 'paper').agg({'score': 'sum'}).collect()[0][0]
	# Divide by number of nodes
	dangling_sum = dangling_sum / num_nodes

	# -------------------------------- #
	# 2. Do a Map Reduce Step to calculate PageRank scores
	
	# Calculate the score that nodes without inbound links would get, from PageRank formula
	uncited_node_score = alpha * dangling_sum + (1-alpha) * random_jump_prob
	# Main Map - Reduce PageRank calculation step. 
	# This proceeds as follows (line by line):
	# 1. First Join outlinks with scores to get a 'table' of the format <citing_paper> <cited_papers> <citing_paper_score>
	# 2. Then we get a list out of the cited papers (using the explode function) and calculate the <citing_paper_score>/<number_of_cited_papers>
	#    We output pairs <cited_paper> <received_score>
	# 3. We then aggregate transerred scores by papers
	# 4. In the resulting dataset we calculate the PageRank Formula. This, however only works for nodes that were cited (i.e., they got a score 
	#    in step 2.
	# 5. We join the result with the previous scores DataFrame and get a result of the form:
	#    <paper> <score> <previous_score>. Since this is a right outer join, it will include papers that were not cited, with a current score of NULL
	# 6. We re-partition the results to balance the calculations and also to benefit the following joins (scores and previous scores DataFrames are
	#    partitioned based on the same key)
	# 7. We fill the null values (i.e., scores of nodes without inbound links) with the previously calculated score for them
	# 8. We add an extra score with the difference between current and previous score
	scores = outlinks_actual.join(scores, 'paper')\
				.select(F.explode(outlinks_actual.cited_papers).alias('paper'), (F.col('score')/F.size(outlinks.cited_papers)).alias('transferred_score'))\
				.groupBy('paper')\
				.agg(F.sum('transferred_score').alias('transferred_score_sum'))\
				.select('paper', (alpha * (F.col('transferred_score_sum')+dangling_sum) + (1-alpha)*random_jump_prob).alias('score'))\
				.join(previous_scores, 'paper', 'right_outer')\
				.repartition(num_partitions, 'paper')\
				.fillna(uncited_node_score, ['score'])\
				.withColumn('score_diff', F.abs( F.col('score') - F.col('previous_score') ) )
	# We should keep the newly calculated scores in memory for further use.
	# If we used the .cache() method, this would lead to a steadily increasing execution plan for the DataFrame
	# since spark wants to be able to recalculate results in case of any failure. However this would lead to 
	# memory errors and the program crashes. To avoid this we checkpoint the data, i.e., we take a snapshot and
	# truncate the lineage data. To make this fast, we use the localCheckpoint() method, instead of the checkpoint()
	# method, which would need time in order to write to the hdfs. Using localCheckpoint() might not be very safe
	# in case of failure, BUT this is a risk we are willing to take here.
	# scores = scores.localCheckpoint()

	# Try reliable checkpointing here!
	if checkpoint_mode == 'dfs':
		# Update checkpoint directory
		if checkpoint_dir.endswith('/'):
			check_dir = checkpoint_dir[:-1]
		else:
			check_dir = checkpoint_dir
		sc.setCheckpointDir(check_dir + '_PageRank_iteration_' +  str(iterations+1))
		# Checkpoint the data
		scores = scores.checkpoint()
		# Clean previous checkpoint directory
		fs.delete( Path(check_dir + '_PageRank_iteration_' +  str(iterations)) )
	elif checkpoint_mode == 'local':
		scores = scores.localCheckpoint()

	# -------------------------------- #
	# 3. Calculate max error
	error = scores.select('score_diff').distinct().agg({'score_diff': 'max'}).collect()[0][0] 

	# -------------------------------- #
	# 4. Do required re-initialisations and update variables

	# Make current scores equal to previous scores
	scores = scores.select('paper','score')
	# Keep only paper and score fields
	previous_scores = scores.select('paper',F.col('score').alias('previous_score'))

	print("--- Iteration " + str(iterations) + " - Time: " + "{:.2f}".format(time.time()-iteration_time) + " (s) - Error: "  + str(error) + " ---")

	# Count iterations
	iterations += 1

print ("\n# ------------------------------------ #\n")
print("Finished score calculations. Preparing classes and normalized scores!")

scores		= scores.repartition(num_partitions, 'paper').cache()
max_score 	= scores.agg({'score': 'max'}).collect()[0]['max(score)']

# Define the top ranges in number of papers
top_001_offset	= int(num_nodes * 0.0001)
top_01_offset	= int(num_nodes * 0.001)
top_1_offset	= int(num_nodes * 0.01)
top_10_offset	= int(num_nodes * 0.1)
# Not needed anymore
# top_20_offset	= int(num_nodes * 0.2)
# ------------------------------------------------------------------------------------------------------ #
# This code is included for small testing datasets. The percentages required may be < 1 for small datasets
top_001_offset = 1 if top_001_offset <= 1 else top_001_offset
top_01_offset = 1 if top_001_offset <= 1 else top_01_offset
top_1_offset = 1 if top_1_offset <= 1 else top_1_offset
top_10_offset = 1 if top_10_offset <= 1 else top_10_offset
# top_20_offset = 1 if top_20_offset <= 1 else top_20_offset
# ------------------------------------------------------------------------------------------------------ #
# Time calculations
start_time = time.time()
# Calculate a running count window of scores, in order to filter out papers w/ scores lower than that of the top 20%
distinct_scores = scores.select(F.col('score')).repartition(num_partitions, 'score').groupBy('score').count()\
				 .withColumn('cumulative', F.sum('count').over(Window.orderBy(F.col('score').desc())))
distinct_scores_count = distinct_scores.count()
print ("Calculated distinct scores num (" + str(distinct_scores_count) + "), time: {} seconds ---".format(time.time() - start_time))

# Time it
start_time = time.time()
# Get scores based on which we find the top 20%, 10%, etc

################################################
# not used anymore 			       #
# distinct_scores = distinct_scores.where(F.col('cumulative') <= top_20_offset ).orderBy(F.col('score').asc()).cache()
# top_20_score  = distinct_scores.first()['score']
################################################
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_10_offset ).orderBy(F.col('score')).cache()
top_10_score  = distinct_scores.first()['score']
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_1_offset ).orderBy(F.col('score')).cache()
top_1_score   = distinct_scores.first()['score']
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_01_offset ).orderBy(F.col('score')).cache()
top_01_score  = distinct_scores.first()['score']
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_001_offset ).orderBy(F.col('score')).cache()
top_001_score = distinct_scores.first()['score']
print ("Calculated minimum scores of the various classes, time: {} seconds ---".format(time.time() - start_time))
# ---------------------------------------------- #
# Notify user of stats
print ("Max score is: " + str(max_score))
print ("Number of papers is: " + str(num_nodes))
# print ("Top 20% score is: " + str(top_20_score))
print ("Top 10% score is: " + str(top_10_score))
print ("Top 1% score is: " + str(top_1_score))
print ("Top 0.1% score is: " + str(top_01_score))
print ("Top 0.01% score is: " + str(top_001_score))

print ("\n\nTable")
print ("Percentage\tScore")
print ("10%\t" + str(top_10_score))
print ("1%\t" + str(top_1_score))
print ("0.1%\t" + str(top_01_score))
print ("0.01%\t" + str(top_001_score))
print ("\n\n")
# ---------------------------------------------- #
# Add 3-scale classes to score dataframe
scores = scores.select('paper', F.col('score').alias('pr'))\
		.withColumn('normalized_pr', F.lit(F.col('pr')/float(max_score)))\
		.withColumn('three_point_class', F.lit('C'))
scores = scores.withColumn('three_point_class', F.when(scores.pr >= top_1_score, F.lit('B')).otherwise(F.col('three_point_class')) )
scores = scores.withColumn('three_point_class', F.when(scores.pr >= top_001_score, F.lit('A')).otherwise(F.col('three_point_class')) )	
scores = scores.select(F.regexp_replace('paper', 'comma_char', ',').alias('doi'), 'pr', 'normalized_pr', 'three_point_class')

# Add six point class to score dataframe
scores = scores.withColumn('five_point_class', F.lit('E'))
# scores = scores.withColumn('six_point_class', F.when(scores.pr >= top_20_score, F.lit('E')).otherwise(F.col('six_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.pr >= top_10_score, F.lit('D')).otherwise(F.col('five_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.pr >= top_1_score, F.lit('C')).otherwise(F.col('five_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.pr >= top_01_score, F.lit('B')).otherwise(F.col('five_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.pr >= top_001_score, F.lit('A')).otherwise(F.col('five_point_class')) )
# ---------------------------------------------- #
print ("Finished! Writing output to file...")
# ---------------------------------------------- #
# Write output in tab delimited format
if mode == 'local':
	input_file_name = input_file.split('/')[-1]
	scores.write.format('csv').mode('overwrite').option('delimiter','\t').option('header',True).save('PR_' + input_file_name.replace('.gz', '') + '_local_a' + str(alpha) + '_error' + str(max_error))
elif mode == 'distributed':
	input_file_name = input_file.split('/')[-1]
	# If we read from a folder, write output in another folder, same directory
	if not input_file_name:
		input_file_name = input_file.split('/')[-2]
		output_prefix = '/'.join(input_file.split('/')[:-2])
	else:
		output_prefix = '/'.join(input_file.split('/')[:-1])	
		
	print("Writing to: " +  output_prefix + "/PR_" + input_file_name.replace('.gz', '') + "_a" + str(alpha) + "_error" + str(max_error))
	scores.write.mode('overwrite').option('delimiter','\t').option('header',True).csv(output_prefix + '/PR_' + input_file_name.replace('.gz', '') + '_a' + str(alpha) + '_error' + str(max_error))

if checkpoint_mode == 'dfs':
        # Clean previous checkpoint directory
        fs.delete( Path(check_dir + '_PageRank_iteration_' +  str(iterations)) )

print ("\n\nFINISHED!\n\n")
###################
# Terminate Spark #
sc.stop()	  #
sys.exit(0)	  #
###################



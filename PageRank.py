#!/usr/bin/python

# Spark-based PageRank calculation using DataFrames.

# Program proceeds as follows:
# We create an dataframe of schema ["paper", "score"]
# and another of the schema ["node", "cited_list"]
# There will be two dataframes of the first type in the progam,
# which will allow us to calculate max score differences
# between consecutive runs. 

# ---------- Imports ------------- #
# NOTE: Use this for python 2
from __future__ import print_function
import sys
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
# Functions to effectively handle data 
# manipulation for DataFrames
import pyspark.sql.functions as F
# Diagnostics
import time
from timeit import default_timer as timer
from datetime import timedelta
# -------------------------------- #

# ----------- Inits -------------- #

# Print output message if not run correctly
###########################################
if len(sys.argv) < 5:
	print ('Usage: pagerank_spark_df.py <input_file> <alpha> <convergence_error> <checkpoint_dir> <(optional)num_partitions> <(optional)checkpointing_mode>')
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
program_start_time = check_time = timer()
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
	sp_session = SparkSession.builder.master('local').appName('Spark PageRank').getOrCreate()
else:
	sp_session = SparkSession.builder.config('spark.cleaner.referenceTracking.cleanCheckpoints', 'true').appName('Spark PageRank').getOrCreate()
	# .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") <--- add this if pagerank fails due to space
# Turn off logging (possible values: OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL
sc.setLogLevel('OFF')

# Set checkpoint dir
sc.setCheckpointDir(checkpoint_dir)
#####################################################################################################
#################################
# Get hdfs file handler object. 
# This solution, although not elegant, does not rely on third party libaries
# Source: https://diogoalexandrefranco.github.io/interacting-with-hdfs-from-pyspark/
URI = sc._gateway.jvm.java.net.URI
Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
# Get master prefix from input file path
master_prefix = "/".join(input_file.split("/")[:5])
fs = FileSystem.get(URI(master_prefix), sc._jsc.hadoopConfiguration())
#################
# Testing
# print ("Master prefix is: " + master_prefix) 
# for remote_file in fs.listStatus( Path( "/".join(input_file.split("/")[:-1])) ):
# 	print( remote_file )
# sys.exit(0)
#################

#################################
#################################
# Set PAGERANK-Specific params
#------------------------------ #
# Set initial error to high value
error 	   = 1000
# Read the initial input
if mode == 'local':
	input_data = sc.textFile('file:' + input_file)
elif mode == 'distributed':
	try:
	    # This should work for zipped files as well, as long as there is a ".gz" in the file name
		print("Reading input from hdfs")
		input_data = sc.textFile(input_file)
	except:
		input_data = sc.textFile(input_file + "/*")
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
scores			= input_data.map(lambda x: (x.split("\t")[0].strip(), x.split("\t")[1].split("|")[-1].strip())).toDF(['paper','score']).repartition(num_partitions, 'paper').cache()
# Duplicate scores to keep track of scores in consecutive iterations
previous_scores		= scores.select('paper',F.col('score').alias('previous_score'))

# Continue initialisation message
print(".", end = '')
sys.stdout.flush()

# Get a DataFrame with pairs of <node, cited_list> (see comment above for input format) 
outlinks = input_data.map(lambda x: (x.split("\t")[0].strip(), x.split("\t")[1].rsplit("|",2)[0].split(","))).toDF(['paper', 'cited_papers'])

# Create a DataFrame with nodes filtered based on whether they cite others or not
outlinks_actual = outlinks.filter(outlinks['cited_papers'][0] != "0").repartition(num_partitions, 'paper').cache()

# Continue intialisation message
print(".", end = '')
sys.stdout.flush()

# Print scores
# distinct_scores = scores.select('score').distinct()
# print ("\nDistinct scores:")
# distinct_scores.show()

# Print previous scores
# distinct_previous_scores = previous_scores.select('previous_score').distinct()
# distinct_previous_scores.show()

# sys.exit(0)

# Collect the dangling nodes from the data - cache it since it will be reused
dangling_nodes = outlinks.filter(outlinks.cited_papers[0] == "0").select('paper').repartition(num_partitions, 'paper').cache()

# Continue intialisation message
print(".", end = '')
sys.stdout.flush()

# Initialise some of the above datasets by invoking actions on them
dangling_nodes.count()
outlinks_actual.count()
scores.count()

# Continue intialisation message
print(".", end = '')
sys.stdout.flush()

# Get materialized versions of those dataframes we need
scores.cache()
dangling_nodes.cache()
outlinks_actual.cache()

# Continue intialisation message
print(". Took: %s seconds!" % (time.time()-initialisation_time))
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
	# print ("\nIteration: ", iterations)

	# -------------------------------- #
	# 1. Calculate PageRank scores transferred by Dangling Nodes 
	# Join by key with scores DataFram and then sum values
	# This returns a single row with a single column, therefore we specifically ask for [0][0] offset
	dangling_sum = dangling_nodes.join(scores, 'paper').agg({"score": "sum"}).collect()[0][0]
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
				.select(F.explode(outlinks_actual.cited_papers).alias('paper'), (scores.score/F.size(outlinks.cited_papers)).alias('transferred_score'))\
				.groupBy('paper')\
				.agg(F.sum('transferred_score').alias('transferred_score_sum'))\
				.select('paper', (alpha* (F.col('transferred_score_sum')+dangling_sum) + (1-alpha)*random_jump_prob).alias('score'))\
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
		if checkpoint_dir.endswith("/"):
			check_dir = checkpoint_dir[:-1]
		else:
			check_dir = checkpoint_dir
		sc.setCheckpointDir(check_dir + "_PageRank_iteration_" +  str(iterations+1))
		# Checkpoint the data
		scores = scores.checkpoint()
		# Clean previous checkpoint directory
		fs.delete( Path(check_dir + "_PageRank_iteration_" +  str(iterations)) )
	elif checkpoint_mode == 'local':
		scores = scores.localCheckpoint()

	# -------------------------------- #
	# 3. Calculate max error
	error = scores.agg({"score_diff": "max"}).collect()[0][0] 

	# -------------------------------- #
	# 4. Do required re-initialisations and update variables

	# Make current scores equal to previous scores
	scores = scores.select('paper','score')
	# Keep only paper and score fields
	previous_scores = scores.select('paper',F.col('score').alias('previous_score'))

	print("--- Iteration " + str(iterations) + " - Time: " + "{:.2f}".format(time.time()-iteration_time) + " (s) - Error: "  + str(error) + " ---")

	# Count iterations
	iterations += 1

print ("\n# ------------------------------------ #")
print("Finished! Writing output to file.")


# Write output in tab delimited format
if mode == 'local':
	input_file_name = input_file.split("/")[-1]
	scores.write.format('csv').mode("overwrite").option("delimiter","\t").save("PR_" + input_file_name.replace(".gz", "") + "_local_a" + str(alpha) + "_error" + str(max_error))
elif mode == 'distributed':
	input_file_name = input_file.split("/")[-1]
	output_prefix = "/".join(input_file.split("/")[3:-1])
	print("Writing to: /" +  output_prefix + "/PR_" + input_file_name.replace(".gz", "") + "_a" + str(alpha) + "_error" + str(max_error))
	scores.write.mode("overwrite").option("delimiter","\t").csv("/" + output_prefix + "/PR_" + input_file_name.replace(".gz", "") + "_a" + str(alpha) + "_error" + str(max_error))

if checkpoint_mode == 'dfs':
        # Clean previous checkpoint directory
        fs.delete( Path(check_dir + "_PageRank_iteration_" +  str(iterations)) )


program_end_time = timer()
print ("\n\nRanking program took: " + str(timedelta(seconds=program_end_time-program_start_time)))	
print ("\n\nFINISHED!\n\n")
###################
# Terminate Spark #
sc.stop()	  	  #
sys.exit(0)	  	  #
###################



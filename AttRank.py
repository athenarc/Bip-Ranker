#!/usr/bin/python

# Implementation of AttRank for Spark using DataFrames

# Program proceeds as follows:
# We create an dataframe of schema ["paper", "score"]
# and another of the schema ["paper", "cited_paper_list"].
# We additionally keep DataFrames with the attention / preferential 
# attachment-based scores as well as the time-based weights

# ---------- Imports ------------- #
# NOTE: Use this for python 2
from __future__ import print_function
import sys

# pyspark-specific imports
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.sql.types import *
# Functions to effectively handle DataFrame data
import pyspark.sql.functions as F

# Diagnostics
import time
# -------------------------------- #

# Print output message if not run correctly
###########################################
if len(sys.argv) < 10:
	print ("\nUsage: AttRank.py <input_file> <alpha> <beta> <gamma> <exponential_factor> <current_year> <start_year> <convergence_error> <checkpoint_dir> <(optional)num_partitions> <optional: checkpointing_mode>\n")
	sys.exit(0)
###########################################

############################################
# Read input parameters
input_file 	= sys.argv[1]
alpha 		= float(sys.argv[2])
beta		= float(sys.argv[3])
gamma		= float(sys.argv[4])
exponential	= float(sys.argv[5])
current_year	= int(sys.argv[6])
start_year	= int(sys.argv[7])
max_error  	= float(sys.argv[8])
checkpoint_dir  = sys.argv[9]
# Set default partitions to 1
try:
        num_partitions = int(sys.argv[10])
except:
        num_partitions = 1
		
# Set checkpoint mode / default checkpoint mode
try:
        checkpoint_mode = sys.argv[11]
except:
        checkpoint_mode = 'dfs'			
# Set the mode by default as dfs. 
if checkpoint_mode not in ['dfs', 'local']:
	checkpoint_mode = 'dfs'
	
# Set the mode by default as local. 
# If data is read from hdfs we switch to cluster
mode = 'local'
if input_file.startswith('hdfs://'):
	mode = 'distributed'
############################################
print("Mode is: " + mode)
initialisation_time = time.time()
# Initialize spark context and spark-specific params
#####################################################################################################
if  mode == 'local':
	conf = SparkConf().setMaster('local').setAppName('Spark AttRank')
else:
	conf = SparkConf().setAppName('Spark AttRank')
# Pass configuration to spark context
sc = SparkContext(conf = conf)
# Session is needed for spark.sql library, in order to use DataFrames
if mode == 'local':
	spark = SparkSession.builder.master('local').appName('Spark AttRank').getOrCreate()
else:
	spark = SparkSession.builder.appName('Spark AttRank').getOrCreate()
# Turn off logging (possible values: OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL
sc.setLogLevel('OFF')

# Set checkpoint dir
sc.setCheckpointDir(checkpoint_dir)
#####################################################################################################
# Define schema for reading the input file
graph_file_schema = StructType([
	StructField('paper', StringType(), False),
	StructField('citation_data', StringType(), False),
	StructField('prev_score', FloatType(), False),
	StructField('pub_year', IntegerType(), False)		
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
master_prefix = "/".join(input_file.split('/')[:5])
fs = FileSystem.get(URI(master_prefix), sc._jsc.hadoopConfiguration())
#################################
# Set PAGERANK-Specific params  #
#------------------------------ #
# Set initial error to high value
error 	   = 1000
# Read the initial input
if mode == 'local':
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file)
elif mode == 'distributed':
	print ("Reading input from HDFS...")
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file).repartition(num_partitions, 'paper')
		
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
print ("Beta: ", beta)
print ("Gamma: ", gamma)
print ("Exponential rho: ", exponential)
print ("Starting year for Recent Attention: ", start_year)
print ("Current Year: ", current_year)
print ("Convergence Error: ", max_error)
print ("Number of nodes: ", num_nodes)
print ("Number of partitions: ", num_partitions) 
print ("Checkpoint mode: " + str(checkpoint_mode))
print ("Checkpoint dir: " + checkpoint_dir)
print ("# ------------------------------------ #\n")

print ("Initialising", end='')
sys.stdout.flush()
##################################################
# Initialise SPARK Data

# Get a DataFrame with pairs of <node, score>.
# Recall the input format to understand what the following map function does:
# <paper> <tab> <cited_papers|num_cited_papers|score> <tab> <previous_score> <tab> <publication_year>
scores = input_data.select('paper', F.split('citation_data', "\|").alias('citation_data') )\
		   .select('paper', F.expr('element_at(citation_data, size(citation_data))').alias('score')).cache()
		   #.select('paper', F.element_at(F.col('citation_data'), F.expr('size(citation_data)') ).alias('score') ).cache()

# Duplicate scores to keep track of scores in consecutive iterations
previous_scores	= scores.select('paper',F.col('score').alias('previous_score'))

# Continue initialisation message
print(".", end = '')
sys.stdout.flush()
# ----------------- #
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

###########################################################
# --> Create a DataFrame with the time-based exponential scores <--
# 1. Get paper-publication year pairs.
paper_years = input_data.select('paper', F.col('pub_year').alias('year')).withColumn('year_fixed', F.when( (F.col('year').cast(IntegerType()) < 1000) | (F.col('year').cast(IntegerType()) > int(current_year)) | (F.col('year') == "\\N"), 0).otherwise(F.col('year')))
paper_years = paper_years.select('paper', F.col('year_fixed').alias('year')).repartition(num_partitions, 'paper').cache()
# 2. Get paper-exponential score-based pairs
paper_exp = paper_years.withColumn('exp_score', F.lit(F.exp(exponential * (current_year+1-paper_years.year) )) ).drop('year')
# 3. Normalize exponential scores so they add to one
exp_score_sum = paper_exp.agg({'exp_score':'sum'}).collect()[0][0]
paper_exp     = paper_exp.select('paper', (paper_exp.exp_score/float(exp_score_sum)).alias('exp_score')).repartition(num_partitions, 'paper').cache()

# Continue Initialisation message
print(".", end = '')
sys.stdout.flush()
###########################################################

###########################################################
# --> Create a DataFrame with the Preferential Attachment/Recent Attention-based scores <--
# 1. Get paper-citations pairs. Keep only papers that are published after start_year and then 
paper_citations	= outlinks_actual\
		  .join(paper_years, 'paper')\
		  .where(paper_years.year >= start_year)\
		  .select(F.explode(outlinks_actual.cited_papers).alias('paper'), F.lit(1).alias('citation') )\
		  .groupBy('paper')\
		  .agg(F.sum('citation').alias('citations_in_range'))

# Paper years won't be needed any more - remove it from memory
paper_years.unpersist(True)

# 2. Get total number of citations made in the specified year range
total_citations_in_range = paper_citations.agg({'citations_in_range':'sum'}).collect()[0][0]
# 3. Calculate preferential attachment probabilities - cache them since they will be reused
paper_attention = paper_citations.select('paper', (F.col('citations_in_range') / total_citations_in_range).alias('attention')).repartition(num_partitions, 'paper').cache()
# Continue Initialisation message
print(".", end = '')
sys.stdout.flush()
###########################################################
# --> Get paper exponential scores and attention scores in a single 'materialized' table <--
vector_scores = paper_attention.join(paper_exp, 'paper', 'right_outer').fillna(0.0, ['attention']).repartition(num_partitions, 'paper').cache()
# vector_scores.count()
# Continue Initialisation message
print(".", end = '')
sys.stdout.flush()
###########################################################
# Continue intialisation message
print(". Took: %s seconds!" % (time.time()-initialisation_time))
print("Starting!")

# print ("Paper attention: " + str(paper_attention.count()))
# print ("vector scores:" + str(vector_scores.count()))


# sys.exit(0)
###########################################################

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
	# Join by key with scores DataFrame and then sum values
	# This returns a single row with a single column, therefore we specifically ask for [0][0] offset
	dangling_sum = dangling_nodes.join(scores, 'paper').agg({'score': 'sum'}).collect()[0][0]
	# Divide by number of nodes
	dangling_sum = dangling_sum / num_nodes
	# print("Dangling sum: ", dangling_sum)


	# TODO: WRITE THIS AGAIN
	# Main Map - Reduce PageRank calculation step. 
	scores = outlinks_actual.join(scores, 'paper')\
				.select(F.explode(outlinks_actual.cited_papers).alias('paper'), (F.col('score')/F.size(outlinks.cited_papers)).alias('transferred_score'))\
				.groupBy('paper')\
				.agg(F.sum('transferred_score').alias('transferred_score_sum'))\
				.join(vector_scores, 'paper', 'right_outer')\
				.repartition(num_partitions, 'paper')\
				.fillna(0.0, ['transferred_score_sum'])\
				.select('paper', (alpha*(F.col('transferred_score_sum')+dangling_sum) + beta*F.col('attention') + gamma*F.col('exp_score') ).alias('score'))\
				.join(previous_scores, 'paper')\
				.withColumn('score_diff', F.abs( F.col('score') - F.col('previous_score') ) )

	# Using localCheckpoint() might not be very safe in case of failure, BUT this is a risk we are willing to take here.
	# We avoid caching mulitple data and growing very large execution plans that would lead to memory errors
	# scores = scores.localCheckpoint()
		
	# Try reliable checkpointing here!
	if checkpoint_mode == 'dfs':
		# Update checkpoint directory
		if checkpoint_dir.endswith('/'):
			check_dir = checkpoint_dir[:-1]
		else:
			check_dir = checkpoint_dir
		sc.setCheckpointDir(check_dir + '_AttRank_iteration_' +  str(iterations+1))
		# Checkpoint the data
		scores = scores.checkpoint()
		# Clean previous checkpoint directory
		fs.delete( Path(check_dir + '_AttRank_iteration_' +  str(iterations)) )
		
	elif checkpoint_mode == 'local':
		scores = scores.localCheckpoint()
	# -------------------------------- #
	# 3. Calculate max error
	error = scores.agg({'score_diff': 'max'}).collect()[0][0] 

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
top_20_offset	= int(num_nodes * 0.2)
# ------------------------------------------------------------------------------------------------------ #
# This code is included for small testing datasets. The percentages required may be < 1 for small datasets
top_001_offset = 1 if top_001_offset <= 1 else top_001_offset
top_01_offset = 1 if top_001_offset <= 1 else top_01_offset
top_1_offset = 1 if top_1_offset <= 1 else top_1_offset
top_10_offset = 1 if top_10_offset <= 1 else top_10_offset
top_20_offset = 1 if top_20_offset <= 1 else top_20_offset
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
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_20_offset ).orderBy(F.col('score').asc()).cache()
top_20_score  = distinct_scores.first()['score']
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
print ("Top 20% score is: " + str(top_20_score))
print ("Top 10% score is: " + str(top_10_score))
print ("Top 1% score is: " + str(top_1_score))
print ("Top 0.1% score is: " + str(top_01_score))
print ("Top 0.01% score is: " + str(top_001_score))
# ---------------------------------------------- #
# Add 3-scale classes to score dataframe
scores = scores.select('paper', F.col('score').alias('attrank'))\
		.withColumn('normalized_attrank', F.lit(F.col('attrank')/float(max_score)))\
		.withColumn('three_point_class', F.lit('C'))
scores = scores.withColumn('three_point_class', F.when(scores.attrank >= top_1_score, F.lit('B')).otherwise(F.col('three_point_class')) )
scores = scores.withColumn('three_point_class', F.when(scores.attrank >= top_001_score, F.lit('A')).otherwise(F.col('three_point_class')) )	
scores = scores.select(F.regexp_replace('paper', 'comma_char', ',').alias('doi'), 'attrank', 'normalized_attrank', 'three_point_class')

# Add six point class to score dataframe
scores = scores.withColumn('six_point_class', F.lit('F'))
scores = scores.withColumn('six_point_class', F.when(scores.attrank >= top_20_score, F.lit('E')).otherwise(F.col('six_point_class')) )
scores = scores.withColumn('six_point_class', F.when(scores.attrank >= top_10_score, F.lit('D')).otherwise(F.col('six_point_class')) )
scores = scores.withColumn('six_point_class', F.when(scores.attrank >= top_1_score, F.lit('C')).otherwise(F.col('six_point_class')) )
scores = scores.withColumn('six_point_class', F.when(scores.attrank >= top_01_score, F.lit('B')).otherwise(F.col('six_point_class')) )
scores = scores.withColumn('six_point_class', F.when(scores.attrank >= top_001_score, F.lit('A')).otherwise(F.col('six_point_class')) )


print ("Finished! Writing output to file.")

# Write output in tab delimited format
if mode == 'local':
	input_file_name = input_file.split('/')[-1]
	scores.write.format('csv').mode('overwrite').option('delimiter','\t').option('header',True).save('AttRank_' + input_file_name + '_local_a' + str(alpha) + '_b' + str(beta) + '_c' + str(gamma) + '_rho' + str(exponential) + '_year' + str(start_year) + '-' + str(current_year) + '_error' + str(max_error))
	print("AttRank_" + input_file_name + "_local_a" + str(alpha) + "_b" + str(beta) + "_c" + str(gamma) + "_rho" + str(exponential) + "_year" + str(start_year) + "-" + str(current_year) + "_error" + str(max_error))
elif mode == 'distributed':
	input_file_name = input_file.split('/')[-1]
	# If we read from a folder, write output in another folder, same directory
	if not input_file_name:
		input_file_name = input_file.split('/')[-2]
		output_prefix = '/'.join(input_file.split('/')[:-2])
	else:
		output_prefix = '/'.join(input_file.split('/')[:-1])	
		
	print("Writing to: " + output_prefix + "/AttRank_" + input_file_name.replace('.gz', '') + '_a' + str(alpha) + '_b' + str(beta) + '_c' + str(gamma) + '_rho' + str(exponential) + '_year' + str(start_year) + '-' + str(current_year) + '_error' + str(max_error))
	scores.write.mode('overwrite').option('delimiter','\t').option('header',True).csv(output_prefix + '/AttRank_' + input_file_name.replace('.gz', '') + '_a' + str(alpha) + '_b' + str(beta) + '_c' + str(gamma) + '_rho' + str(exponential) + '_year' + str(start_year) + '-' + str(current_year) + '_error' + str(max_error))

if checkpoint_mode == 'dfs':
        # Clean previous checkpoint directory
        fs.delete( Path(check_dir + "_AttRank_iteration_" +  str(iterations)) )

print ("\n\nFINISHED!\n\n")
###################
# Terminate Spark #
sc.stop()	  #
sys.exit(0)	  #
###################


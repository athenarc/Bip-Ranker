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
from pyspark.sql.types import IntegerType
# -------------------------------- #

# Print output message if not run correctly
###########################################
if len(sys.argv) < 10:
	print ('\nUsage: AttRank.py <input_file> <alpha> <beta> <gamma> <exponential_factor> <current_year> <start_year> <convergence_error> <checkpoint_dir> <(optional)num_partitions> <optional: checkpointing_mode>\n')
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
checkpoint_dir = sys.argv[9]
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
program_start_time = check_time = timer()
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
	sp_session = SparkSession.builder.master('local').appName('Spark AttRank').getOrCreate()
else:
	sp_session = SparkSession.builder.appName('Spark AttRank').getOrCreate()
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
# Set PAGERANK-Specific params  #
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
scores			= input_data.map(lambda x: (x.split()[0].strip(), x.split()[1].split("|")[-1].strip())).toDF(['paper','score']).repartition(num_partitions, 'paper').cache()
# Duplicate scores to keep track of scores in consecutive iterations
previous_scores		= scores.select('paper',F.col('score').alias('previous_score'))

# Continue initialisation message
print(".", end = '')
sys.stdout.flush()

# Get a DataFrame with pairs of <node, cited_list> (see comment above for input format) 
outlinks = input_data.map(lambda x: (x.split()[0].strip(), x.split()[1].rsplit("|",2)[0].split(","))).toDF(['paper', 'cited_papers'])

# Create a DataFrame with nodes filtered based on whether they cite others or not
outlinks_actual = outlinks.filter(outlinks['cited_papers'][0] != "0").repartition(num_partitions, 'paper').cache()

# Continue intialisation message
print(".", end = '')
sys.stdout.flush()

# Collect the dangling nodes from the data - cache it since it will be reused
dangling_nodes = outlinks.filter(outlinks.cited_papers[0] == "0").select('paper').repartition(num_partitions, 'paper').cache()

# Continue intialisation message
print(".", end = '')
sys.stdout.flush()

###########################################################
# --> Create a DataFrame with the time-based exponential scores <--
# 1. Get paper-publication year pairs.
paper_years	= input_data.map(lambda x: (x.split()[0].strip(), x.split()[-1])).toDF(['paper','year']).repartition(num_partitions, 'paper')\
			    .withColumn('year_fixed', F.when( (F.col('year').cast(IntegerType()) < 1000) | (F.col('year').cast(IntegerType()) > int(current_year)) | (F.col('year') == "\\N"), 0).otherwise(F.col('year')))
# We have a case of erroneous years when translating openaire IDs. 
# We insert a year validation here (has to be in 1000-2021)
# paper_years = paper_years.select('paper', 'year').withColumn('year_fixed', F.col() ).cache()
#################
# TESTING
# paper_years.groupBy("year_fixed").count().sort(F.col('count').asc()).show(4000, False)
# sys.exit(0)
#################
paper_years = paper_years.select('paper', F.col('year_fixed').alias('year')).cache()
# 2. Get paper-exponential score-based pairs
paper_exp 	= paper_years.withColumn('exp_score', F.lit(F.exp(exponential * (current_year+1-paper_years.year) )) ).select('paper', 'exp_score')
# 3. Normalize exponential scores so they add to one
exp_score_sum	= paper_exp.agg({'exp_score':'sum'}).collect()[0][0]
paper_exp	= paper_exp.select('paper', (paper_exp.exp_score/float(exp_score_sum)).alias('exp_score')).repartition(num_partitions, 'paper').cache()
# Paper years won't be needed any more - remove it from memory
paper_years.unpersist()
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
vector_scores.count()
# Continue Initialisation message
print(".", end = '')
sys.stdout.flush()
###########################################################

# Continue intialisation message
print(". Took: %s seconds!" % (time.time()-initialisation_time))
print("Starting!")
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
	dangling_sum = dangling_nodes.join(scores, 'paper').agg({"score": "sum"}).collect()[0][0]
	# Divide by number of nodes
	dangling_sum = dangling_sum / num_nodes
	# print("Dangling sum: ", dangling_sum)


	# TODO: WRITE THIS AGAIN
	# Main Map - Reduce PageRank calculation step. 
	scores = outlinks_actual.join(scores, 'paper')\
				.select(F.explode(outlinks_actual.cited_papers).alias('paper'), (scores.score/F.size(outlinks.cited_papers)).alias('transferred_score'))\
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
	
	# TESTING
	# SORT RESULT BASED ON DIFFERENCE AND DISPLAY
	# scores.sort(scores.score_diff.desc()).show(20, False)
	
	# Try reliable checkpointing here!
	if checkpoint_mode == 'dfs':
		# Update checkpoint directory
		if checkpoint_dir.endswith("/"):
			check_dir = checkpoint_dir[:-1]
		else:
			check_dir = checkpoint_dir
		sc.setCheckpointDir(check_dir + "_AttRank_iteration_" +  str(iterations+1))
		# Checkpoint the data
		scores = scores.checkpoint()
		# Clean previous checkpoint directory
		fs.delete( Path(check_dir + "_AttRank_iteration_" +  str(iterations)) )
		
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


print ("\n# ------------------------------------ #\n")
print("Finished! Writing output to file.")


# Write output in tab delimited format
if mode == 'local':
	input_file_name = input_file.split("/")[-1]
	scores.write.format('csv').mode("overwrite").option("delimiter","\t").save("AttRank_" + input_file_name + "_local_a" + str(alpha) + "_b" + str(beta) + "_g" + str(gamma) + "_rho" + str(exponential) + "_year" + str(start_year) + "-" + str(current_year) + "_error" + str(max_error))
	print("AttRank_" + input_file_name + "_local_a" + str(alpha) + "_b" + str(beta) + "_g" + str(gamma) + "_rho" + str(exponential) + "_year" + str(start_year) + "-" + str(current_year) + "_error" + str(max_error))
elif mode == 'distributed':
	input_file_name = input_file.split("/")[-1]
	output_prefix = "/".join(input_file.split("/")[3:-1])
	print("Writing to: /" + output_prefix + "/AttRank_" + input_file_name.replace(".gz", "") + "_a" + str(alpha) + "_b" + str(beta) + "_g" + str(gamma) + "_rho" + str(exponential) + "_year" + str(start_year) + "-" + str(current_year) + "_error" + str(max_error))
	scores.write.mode("overwrite").option("delimiter","\t").csv("/" + output_prefix + "/AttRank_" + input_file_name.replace(".gz", "") + "_a" + str(alpha) + "_b" + str(beta) + "_g" + str(gamma) + "_rho" + str(exponential) + "_year" + str(start_year) + "-" + str(current_year) + "_error" + str(max_error))

if checkpoint_mode == 'dfs':
        # Clean previous checkpoint directory
        fs.delete( Path(check_dir + "_AttRank_iteration_" +  str(iterations)) )

program_end_time = timer()
print ("\n\nRanking program took: " + str(timedelta(seconds=program_end_time-program_start_time)))	
print ("\n\nFINISHED!\n\n")
###################
# Terminate Spark #
sc.stop()	  #
sys.exit(0)	  #
###################


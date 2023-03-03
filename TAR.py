#!/usr/bin/python

# Implementation of Time-Aware Ranking for Spark using DataFrames

# Program proceeds as follows:
# We create an dataframe of schema ["paper", "score"]
# and another of the schema ["paper", "cited_paper_list"].

# TODO: Use this for python 2
from __future__ import print_function
# ---------- Imports ------------- #
import sys

# Pyspark specific things
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
# Functions to DataFrame data
import pyspark.sql.functions as F
from pyspark.sql import Window

# Diagnostics
import time
# -------------------------------- #

# Print output message if not run correctly
###########################################
if len(sys.argv) < 7:
	print ("\nUsage: ./timeawarerank.py <input_file> <gamma> <current_year> <mode:RAM|ECM> <num_partitions> <checkpoint_dir> <optional: checkpoint_mode> <optional:alpha> <optional: max_error>\n")
	sys.exit(0)
###########################################

############################################
# Read input parameters
input_file 	= sys.argv[1]
gamma		= float(sys.argv[2])
current_year	= int(sys.argv[3])
tar_type	= sys.argv[4].lower()
# Set default partitions to 1
try:
	num_partitions = int(sys.argv[5])
except:
	num_partitions = 1
	print ("For num partitions I found: " + str(sys.argv[5]))

try:
	checkpoint_dir = sys.argv[6]
except:
	checkpoint_dir = 'check'
	
# If it's not specified whether to use RAM or ECM, we default to RAM
if tar_type not in ['ram','ecm']:
	tar_type = 'ram'
	
# Set checkpoint mode / default checkpoint mode
try:
        checkpoint_mode = sys.argv[7]
except:
        checkpoint_mode = 'dfs'		
		
# Read ECM parameter
if tar_type != 'ram':
	try:
		alpha 		= float(sys.argv[8])
		max_error  	= float(sys.argv[9])
	except:
		print ("\nUsage: ./timeawarerank.py <input_file> <gamma> <current_year> <mode:RAM|ECM> <num_partitions> <checkpoint_dir> <optional: checkpoint_mode> <optional:alpha> <optional: max_error> \n")
		sys.exit(0)
# Set the mode by default as local. 
# If data is read from hdfs we switch to cluster
mode = 'local'
if input_file.startswith('hdfs://'):
	mode = 'distributed'
	
# Set the mode by default as dfs. 
if checkpoint_mode not in ['dfs', 'local']:
	checkpoint_mode = 'dfs'	
############################################
# Calculate initialization time
initialisation_time = time.time()
# Initialize spark context and spark-specific params
#####################################################################################################
if  mode == 'local':
	conf = SparkConf().setMaster('local').setAppName('Spark Time-Aware Ranking')
else:
	conf = SparkConf().setAppName('Spark Time-Aware Ranking')
# Pass configuration to spark context
sc = SparkContext(conf = conf)
# Session is needed for spark.sql library, in order to use DataFrames
if mode == 'local':
	spark = SparkSession.builder.master('local').appName('Spark Time-Aware Ranking').getOrCreate()
else:
	spark = SparkSession.builder.config('spark.cleaner.referenceTracking.cleanCheckpoints', 'true').appName('Spark Time-Aware Ranking').getOrCreate()
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
# Set CiteRank-Specific params  #
#------------------------------ #
# Set initial error to high value
error 	   = 1000
# Read the initial input
if mode == 'local':
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file)
elif mode == 'distributed':
	print("\n\nReading input from hdfs\n\n")
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file).repartition(num_partitions, 'paper')
	
# Get number of nodes (one node-record per line)
num_nodes = float(input_data.count())
# Count iterations
iterations = 1
##################################
# Display CR parameterisation
print ("\n# ------------------------------------ #")
print ("Input: " + input_file)
print ("Gamma: " + str(gamma))
print ("Current Year: " + str(current_year))
print ("Ranking mode: " + tar_type.upper())
if tar_type == 'ecm':
	print ("Alpha: " + str(alpha))
	print ("Convergence Error: " + str(max_error))
print ("Number of nodes: " + str(num_nodes))
print ("Number of partitions: " + str(num_partitions) )
print ("Checkpoint mode: " + str(checkpoint_mode))
print ("Checkpoint dir: " + checkpoint_dir)
print ("# ------------------------------------ #\n")

print ("Initialising", end='')
sys.stdout.flush()
##################################################
# Initialise SPARK Data

# Get a DataFrame with pairs of <node, year>.
# Recall the input format to understand what the following map function does:
# <paper> <tab> <cited_papers|num_cited_papers|score> <tab> <previous_score> <tab> <publication_year>
paper_years = input_data.select('paper', 'pub_year').withColumn('year_fixed', F.when( (F.col('pub_year').cast(IntegerType()) < 1000) | (F.col('pub_year').cast(IntegerType()) > int(current_year)) | (F.col('pub_year') == "\\N"), 0).otherwise(F.col('pub_year')))

# We have a case of erroneous years when translating openaire IDs. 
# We insert a year validation here (has to be in 1000-2021)
paper_years = paper_years.select('paper', F.col('year_fixed').alias('year')).repartition(num_partitions, 'paper').cache()

# Get a DataFrame with pairs of <node, cited_list> (see comment above for input format) 
# Keep only papers that DO cite other papers
outlinks = input_data.select("paper", F.split("citation_data", "\|").alias("cited_papers"), "pub_year")\
		     .select("paper", "cited_papers", F.expr("size(cited_papers)-2").alias("cited_paper_size"), "pub_year")\
		     .select("paper", F.expr("slice(cited_papers, 1, cited_paper_size)").alias("cited_papers"), "pub_year")\
		     .select("paper", F.array_join("cited_papers", "|").alias("cited_papers"), "pub_year")\
		     .select("paper", F.split("cited_papers", ",").alias("cited_papers"), "pub_year").repartition(num_partitions, "paper").cache()

print(".", end = '')
sys.stdout.flush()

# If we do ECM calculations, we need to additionally initialise
# TODO: write this
if tar_type == 'ecm':
	scores = input_data.select('paper', F.split('citation_data', "\|").alias('citation_data'), F.col('pub_year').alias('year') )\
			   .select('paper', F.expr('element_at(citation_data, size(citation_data))').alias('score')).cache()
			   #.select('paper', F.element_at("citation_data", F.expr("size(citation_data)") ).alias('score'), 'year' ).cache()
			   
	previous_scores = scores.select('paper', F.col('score').alias('previous_score'))
###########################################################
# Continue intialisation message
print(". Took: %s seconds!" % (time.time()-initialisation_time))
print("Starting!")
###########################################################

# IF we use RAM, only one calculation is required....
if tar_type == 'ram':
	scores = outlinks.join(paper_years, 'paper')\
			 .select(F.explode(outlinks.cited_papers).alias('paper'), (gamma ** (current_year - F.col('year')) ).alias('transferred_score'))\
			 .groupBy('paper')\
			 .agg(F.sum('transferred_score').alias('score'))\
			 .join(paper_years, 'paper', 'right_outer')\
			 .fillna(0.0, ['score'])\
			 .select('paper', 'score')

# .... Otherwise we have to iteratively calculate scores
else:

	# Calculate the initial score - since in each iteration the we must use the previous score we do the initial step separately
	scores = outlinks.join(scores, 'paper')\
			 .select(F.explode(outlinks.cited_papers).alias('paper'), (alpha*(gamma ** (current_year - F.col('year')))).alias('transferred_score'), 'year')\
			 .groupBy('paper')\
			 .agg(F.sum('transferred_score').alias('score'))\
			 .join(previous_scores, 'paper', 'right_outer')\
			 .select('paper','score','year').repartition(num_partitions, 'paper')\
			 .fillna(0.0, ['score'])

	previous_scores = scores.select('paper', F.col('score').alias('previous_score'), 'year')
	# Loop until error is less than the threshold given
	while error >= max_error:

		#######################################################
		# Time the iteration
		iteration_time = time.time()
		# Notify user
		print ("\n# ------------------------------------ #\n")
		#######################################################

		# .select('paper', (F.col('score') * (alpha ** (iterations)) ).alias('running_ecm'))\
		# ---------------------------------------------------------------------- #
		# 1. Do the basic map-reduce step to calculate scores
		scores = outlinks.join(scores, 'paper')\
				 .select(F.explode(outlinks.cited_papers).alias('paper'), (alpha*(gamma ** (current_year - F.col('year')))*F.col('score')).alias('transferred_score'))\
				 .groupBy('paper')\
				 .agg(F.sum('transferred_score').alias('running_ecm'))\
				 .join(previous_scores, 'paper', 'right_outer')\
				 .repartition(num_partitions, 'paper')\
				 .fillna(0.0, 'running_ecm')\
				 .select('paper', 'running_ecm', 'previous_score', (F.col('running_ecm')+F.col('previous_score')).alias('ecm'), 'year')

		# Keep a local checkpoint of the scores since they will be used again in the next iteration
		if checkpoint_mode == 'dfs':
			# Update checkpoint directory
			if checkpoint_dir.endswith("/"):
				check_dir = checkpoint_dir[:-1]
			else:
				check_dir = checkpoint_dir
			sc.setCheckpointDir(check_dir + "_TAR_ECM_iteration_" +  str(iterations+1))
			# Checkpoint the data
			scores = scores.checkpoint()
			# Clean previous checkpoint directory
			fs.delete( Path(checkpoint_dir + "_TAR_ECM_iteration_" +  str(iterations)) )
		elif checkpoint_mode == 'local':
			scores = scores.localCheckpoint()
		# ---------------------------------------------------------------------- #
		# 2. Calculate error between successive scores -> since scores are incrementally calculated 
		# by adding an extra part in each iteration, we only need to find the maximum added score
		error = scores.agg({'running_ecm':'max'}).collect()[0][0]

		# ---------------------------------------------------------------------- #
		# 3. Re-initialise variables and DataFrames

		# Keep only running citerank part in scores and keep the previous accumulated citerank values as the previous score
		previous_scores = scores.select('paper', F.col('ecm').alias('previous_score'), 'year')
		scores 		= scores.select('paper', F.col('running_ecm').alias('score'), 'year')
		# ---------------------------------------------------------------------- #
		print("--- Iteration " + str(iterations) + " - Time: " + "{:.2f}".format(time.time()-iteration_time) + " (s) - Error: "  + str(error) + " ---")
		# Count iterations
		iterations += 1


print ("\n# ------------------------------------ #\n")
print("Finished preparations... Calculating scores and classes...\n")

# Time it
start_time = time.time()
scores		= scores.repartition(num_partitions, 'paper').cache()
max_score 	= scores.select('score').distinct().agg({'score': 'max'}).collect()[0]['max(score)']
print ("Got max score:" + str(max_score) + " - Took {} seconds".format(time.time() - start_time) + " to get here from initial file read (this is the first transformation)")


# Define the top ranges in number of papers
top_001_offset	= int(num_nodes * 0.0001)
top_01_offset	= int(num_nodes * 0.001)
top_1_offset	= int(num_nodes * 0.01)
top_10_offset	= int(num_nodes * 0.1)
# top_20_offset	= int(num_nodes * 0.2)
# ------------------------------------------------------------------------------------------------------ #
# This code is included for small testing datasets. The percentages required may be < 1 for small datasets
top_001_offset = 1 if top_001_offset <= 1 else top_001_offset
top_01_offset = 1 if top_001_offset <= 1 else top_01_offset
top_1_offset = 1 if top_1_offset <= 1 else top_1_offset
top_10_offset = 1 if top_10_offset <= 1 else top_10_offset
# top_20_offset = 1 if top_20_offset <= 1 else top_20_offset
# ------------------------------------------------------------------------------------------------------ #	
# Get distinct scores and find score values for top 20%, 10% etc...

# Time it
start_time = time.time()
# Calculate a running count window of scores, in order to filter out papers w/ scores lower than that of the top 20%
distinct_scores = scores.select(F.col('score')).repartition(num_partitions, 'score').groupBy('score').count()\
				 .withColumn('cumulative', F.sum('count').over(Window.orderBy(F.col('score').desc())))
distinct_scores_count = distinct_scores.count()
print ("Calculated distinct scores num (" + str(distinct_scores_count) + "), time: {} seconds ---".format(time.time() - start_time))

# Time it
start_time = time.time()
# Get scores based on which we find the top 20%, 10%, etc
# distinct_scores = distinct_scores.where(F.col('cumulative') <= top_20_offset ).orderBy(F.col('score').asc()).cache()
# top_20_score  = distinct_scores.first()['score']
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
# Inform the user of statistics
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
scores = scores.select('paper', F.col('score').alias('ram'))\
		.withColumn('normalized_ram', F.lit(F.col('ram')/float(max_score)))\
		.withColumn('three_point_class', F.lit('C'))
scores = scores.withColumn('three_point_class', F.when(scores.ram >= top_1_score, F.lit('B')).otherwise(F.col('three_point_class')) )
scores = scores.withColumn('three_point_class', F.when(scores.ram >= top_001_score, F.lit('A')).otherwise(F.col('three_point_class')) )	
scores = scores.select(F.regexp_replace('paper', 'comma_char', ',').alias('doi'), 'ram', 'normalized_ram', 'three_point_class')

# Add six point class to score dataframe
scores = scores.withColumn('five_point_class', F.lit('E'))
# scores = scores.withColumn('five_point_class', F.when(scores.ram >= top_20_score, F.lit('E')).otherwise(F.col('five_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.ram >= top_10_score, F.lit('D')).otherwise(F.col('five_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.ram >= top_1_score, F.lit('C')).otherwise(F.col('five_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.ram >= top_01_score, F.lit('B')).otherwise(F.col('five_point_class')) )
scores = scores.withColumn('five_point_class', F.when(scores.ram >= top_001_score, F.lit('A')).otherwise(F.col('five_point_class')) )


print ("Finished! Writing output to file.")

# Write output in tab delimited format in LOCAL file system
if mode == 'local':
	input_file_name = input_file.split("/")[-1]
	# RAM
	if tar_type.lower() == 'ram':
		scores.write.format('csv').mode('overwrite').option('delimiter','\t').option('header',True).save('RAM_' + input_file_name + '_local_c' + str(gamma) + '_year' + str(current_year))
		print("RAM_" + input_file_name + "_local_c" + str(gamma) + "_year" + str(current_year))
	# ECM	
	else:
		previous_scores.write.format('csv').mode('overwrite').option('delimiter','\t').option('header', True).save('ECM_' + input_file_name + '_local_c' + str(gamma) + '_year' + str(current_year) + '_a' + str(alpha) + '_error' + str(max_error))
		print("ECM_" + input_file_name + "_local_c" + str(gamma) + "_year" + str(current_year) + "_a" + str(alpha) + "_error" + str(max_error))
# Output in HDFS
elif mode == 'distributed':
	input_file_name = input_file.split("/")[-1]
	# If we read from a folder, write output in another folder, same directory
	if not input_file_name:
		input_file_name = input_file.split('/')[-2]
		output_prefix = '/'.join(input_file.split('/')[:-2])
	else:
		output_prefix = '/'.join(input_file.split('/')[:-1])
	# RAM
	if tar_type.lower() == 'ram':
		print("Writing to: " + output_prefix + "/RAM_" + input_file_name.replace(".gz", "") + "_c" + str(gamma) + "_year" + str(current_year))
		scores.write.mode('overwrite').option('delimiter','\t').option('header',True).csv(output_prefix + '/RAM_' + input_file_name.replace('.gz', '') + '_c' + str(gamma) + '_year' + str(current_year))
	# ECM	
	else:
		print("Writing to: " + output_prefix + "/ECM_" + input_file_name.replace('.gz', '') + "_c" + str(gamma) + "_year" + str(current_year) + "_a" + str(alpha) + "_error" + str(max_error))
		previous_scores.write.mode('overwrite').option('delimiter','\t').option('header',True).csv(output_prefix + '/ECM_' + input_file_name.replace('.gz', '') + '_c' + str(gamma) + '_year' + str(current_year)  + '_a' + str(alpha) + '_error' + str(max_error))

print ("\n\nDone!\n\n")	
###################
# Terminate Spark #
sc.stop()	  #
sys.exit(0)	  #
###################

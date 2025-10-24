#!/usr/bin/python

# Implementation of Indicator counting citations up to X
# years after publication - or all - depending on input

# Program proceeds as follows:
# We create an dataframe of schema ["citing_paper", "cited_paper", "citing_paper_year"]
# the first and last columns we read from file, the second is created by a split of the
# cited list. 
# We join this schema with the publication years based on the CITED paper to get a 
# dataframe of the form ["citing_paper", "cited_paper", "citing_paper_year", "cited_paper_year"]

# We perform an aggregation of 1's (i.e. 1 citation) for all records where citing_paper_year - cited_paper_year <= 3.

# After calculating the citation counts we also normalize the scores by dividing each score by the maximum score calculated.
# Further we add indicators of three and six classed based on the top 20% 10% 1% 0.1% 0.01% or the remaining 80%

#---------- Imports ------------- #
import sys

# Import pyspark-specific libraries
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
# This code imports structured types
from pyspark.sql.types import *
from pyspark.sql import Window
# Functions to process DataFrame columns
import pyspark.sql.functions as F

# Measuring Time-Diagnostics
import time

# -------------------------------- #
# Print output message if not run correctly
###########################################
if len(sys.argv) < 2:
	print ("\nUsage: CC.py <input_file> <(optional)num_partitions> <(optional) prefix> <(optional) limit_year>\n")
	sys.exit(0)
###########################################

############################################
# Read input parameters
input_file  = sys.argv[1]

# Set default partitions to 1
try:
	num_partitions = int(sys.argv[2])
except:
	num_partitions = 1
try:
	prefix = sys.argv[3] + "_"
except:
	prefix = "CC_"
try:
	 limit_year = int(sys.argv[4])
except:
	 limit_year = None
	 
# Set the mode by default as local. 
# If data is read from hdfs we switch to cluster
mode = 'local'
if input_file.startswith('hdfs://'):
	mode = 'distributed'
#####################################################################################################
# Create spark session & context - these are entry points to spark
program_name = 'Spark CC' if not  limit_year else ('Spark ' + str( limit_year) + '-year CC')
# Initialize spark context and spark-specific params
if  mode == 'local':
	conf = SparkConf().setMaster('local').setAppName(program_name)
else:
	conf = SparkConf().setAppName(program_name)
# This MAY be required for the right_outer join when running CC with year offset
conf.set('spark.sql.crossJoin.enabled', True)
# Pass configuration to spark context
sc = SparkContext(conf = conf)
# Session is needed for spark.sql library, in order to use DataFrames
if mode == 'local':
	spark = SparkSession.builder.master('local').appName(program_name).getOrCreate()
else:
	spark = SparkSession.builder.appName(program_name).getOrCreate()
# Turn off logging (possible values: OFF, FATAL, ERROR, WARN, INFO, DEBUG, TRACE, ALL
sc.setLogLevel('OFF')
#####################################################################################################
# Define schema for reading the input file
graph_file_schema = StructType([
	StructField('paper', StringType(), False),
	StructField('citation_data', StringType(), False),
	StructField('prev_score', FloatType(), False),
	StructField('pub_year', IntegerType(), False)		
	])
#####################################################################################################
# Read the initial input
if mode == 'local':
	# Use spark session with schema instead of spark context and text file (this should spead up reading the file)
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file)
elif mode == 'distributed':
	print("\n\nReading input from hdfs\n\n")
	# Use spark session with schema instead of spark context and text file (this should spead up reading the file)
	input_data = spark.read.schema(graph_file_schema).option('delimiter', '\t').csv(input_file).repartition(num_partitions, "paper")	
#####################################################################################################	
# Time initialization
initialisation_time = time.time()
# Print out info messages about the program's parameters
print ("Mode is: " + mode)
print ("Num Partitions: " + str(num_partitions))
print ("Limit year: " + str(limit_year))
print ("\n\n")
# Initialise SPARK Data
print("Planning citation data calculation...")
# ------------------ #
# Create the outlinks by selecting and splitting the appropriate fields - split on "|" character and remove last two fields. Then join again on "|" because it may be part of a doi
outlinks = input_data.select('paper', F.split('citation_data', "\|").alias('cited_papers'), 'pub_year')\
		     .select('paper', 'cited_papers', F.expr('size(cited_papers)-2').alias("cited_paper_size"), 'pub_year')\
		     .select('paper', F.expr("slice(cited_papers, 1, cited_paper_size)").alias('cited_papers'), 'pub_year')\
		     .select('paper', F.array_join('cited_papers', '|').alias('cited_papers'), 'pub_year')\
		     .select('paper', F.split('cited_papers', ',').alias('cited_papers'), 'pub_year').repartition(num_partitions, 'pub_year').cache()

# Create a dataframe with nodes filtered based on whether they cite others or not. Here we keep those that make citations (i.e., remove dangling nodes)
print ("Planning removal of dangling nodes...")
outlinks_actual = outlinks.filter(outlinks['cited_papers'][0] != '0')\
			  .select('paper', F.explode(F.col('cited_papers')).alias('cited_paper') , F.col('pub_year')).repartition(num_partitions, "paper").cache()

# If offset year is given, we need to perform some filtering of citations based on pub year. Proceed by normally calculating the 3-year based CC
if  limit_year:
	# We now need to filter out those records where citing year - cited year >  limit_year
	# a. join again with years, based on cited paper year - create a clone of the initial dataframe, because otherwise there will be an error due to similar column names
	print ("Gathering years of cited papers...")
	cited_paper_years = outlinks.select('paper', F.col('pub_year').alias('cited_paper_year')).withColumnRenamed('paper', 'cited_paper').repartition(num_partitions, 'cited_paper')
	# Since here outlinks_actual is joined on cited paper, we need to repartition it
	valid_citations   = outlinks_actual.repartition(num_partitions, 'cited_paper').join(cited_paper_years, outlinks_actual.cited_paper == cited_paper_years.cited_paper)\
			    .select(outlinks_actual.paper, 
				    cited_paper_years.cited_paper, 
				    outlinks_actual.pub_year.alias('citing_paper_year'), 
				    cited_paper_years.cited_paper_year)\
			    .repartition(num_partitions, 'paper')	
					 
	# b. Filter out those where citing paper year > cited paper year + 3
	print ("Filtering out citations based on pub year difference...")
	valid_citations = valid_citations.filter(valid_citations['citing_paper_year']-valid_citations['cited_paper_year'] <=  limit_year).repartition(num_partitions, 'paper').cache()
# Do nothing if no limit year was specified. For uniformity reasons we set the valid citations variable to point to outlinks_actual
else:
	valid_citations = outlinks_actual

# Group by cited_paper and get counts
print("Preparing count of citations...")
valid_citations = valid_citations.repartition(num_partitions, 'cited_paper').groupBy('cited_paper').count().repartition(num_partitions, 'cited_paper')

# Add papers which aren't cited
print("Planning addition of dangling nodes...")
# Join with papers that aren't cited
valid_citations = valid_citations.join(outlinks.select('paper'), outlinks.paper == valid_citations.cited_paper, 'right_outer')\
				.select('paper', 'count')\
				.fillna(0).repartition(num_partitions, 'paper').cache()

print ("\n# ------------------------------------ #\n")
print("Finished planning calculations. Proceeding to calculation of scores and classes...\n")

# Time it
start_time = time.time()
max_score  = valid_citations.select('count').repartition(num_partitions).distinct().agg({'count': 'max'}).collect()[0]['max(count)']
print ("Got max score:" + str(max_score) + " - Took {} seconds".format(time.time() - start_time) + " to get here from initial file read (this is the first transformation)")
 
# Time it
start_time = time.time()
num_papers = valid_citations.count()
print ("Got num. of papers:" + str(num_papers) + " - Took {} seconds".format(time.time() - start_time)) 

# Define the top ranges in number of papers
top_001_offset	= int(num_papers * 0.0001)
top_01_offset	= int(num_papers * 0.001)
top_1_offset	= int(num_papers * 0.01)
top_10_offset	= int(num_papers * 0.1)
# top_20_offset	= int(num_papers * 0.2)
# ------------------------------------------------------------------------------------------------------ #
# This code is included for small testing datasets. The percentages required may be < 1 for small datasets
top_001_offset = 1 if top_001_offset <= 1 else top_001_offset
top_01_offset = 1 if top_001_offset <= 1 else top_01_offset
top_1_offset = 1 if top_1_offset <= 1 else top_1_offset
top_10_offset = 1 if top_10_offset <= 1 else top_10_offset
# top_20_offset = 1 if top_20_offset <= 1 else top_20_offset
# ------------------------------------------------------------------------------------------------------ #
	
# Time it
start_time = time.time()
# Calculate a running count window of scores, in order to filter out papers w/ scores lower than that of the top 20%
distinct_scores = valid_citations.select(F.col('count').alias('cc')).repartition(num_partitions, 'cc').groupBy('cc').count()\
				 .withColumn('cumulative', F.sum('count').over(Window.orderBy(F.col('cc').desc())))
distinct_scores_count = distinct_scores.count()
print ("Calculated distinct scores num (" + str(distinct_scores_count) + "), time: {} seconds ---".format(time.time() - start_time))

# Time it
start_time = time.time()
# Get scores based on which we find the top 20%, 10%, etc
# distinct_scores = distinct_scores.where(F.col('cumulative') <= top_20_offset ).orderBy(F.col('cc').asc()).cache()
# top_20_score  = distinct_scores.first()['cc']
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_10_offset ).orderBy(F.col('cc')).cache()
top_10_score  = distinct_scores.first()['cc']
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_1_offset ).orderBy(F.col('cc')).cache()
top_1_score   = distinct_scores.first()['cc']
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_01_offset ).orderBy(F.col('cc')).cache()
top_01_score  = distinct_scores.first()['cc']
distinct_scores = distinct_scores.where(F.col('cumulative') <= top_001_offset ).orderBy(F.col('cc')).cache()
top_001_score = distinct_scores.first()['cc']
print ("Calculated minimum scores of the various classes, time: {} seconds ---".format(time.time() - start_time))

# ---------------------------------------------- #
# Inform the user of statistics
print ("Max score is: " + str(max_score))
print ("Number of papers is: " + str(num_papers))

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
# ---------------------------------------------- #

# ---------------------------------------------- #
# Add classes to final dataframe
column_name = 'cc'
if limit_year:
	column_name = str(limit_year) + '-cc'

# Add 3-scale classes to score dataframe
valid_citations = valid_citations.select('paper', F.col('count').alias(column_name))\
		.withColumn('normalized_' + column_name, F.lit(F.col(column_name)/float(max_score)))\
		.withColumn('three_point_class', F.lit('C'))
valid_citations = valid_citations.withColumn('three_point_class', F.when(F.col(column_name) >= top_1_score, F.lit('B')).otherwise(F.col('three_point_class')) )
valid_citations = valid_citations.withColumn('three_point_class', F.when(F.col(column_name) >= top_001_score, F.lit('A')).otherwise(F.col('three_point_class')) )	
valid_citations = valid_citations.select(F.regexp_replace('paper', 'comma_char', ',').alias('doi'), column_name, 'normalized_' + column_name, 'three_point_class')

# Add six point class to score dataframe
valid_citations = valid_citations.withColumn('five_point_class', F.lit('E'))
# valid_citations = valid_citations.withColumn('five_point_class', F.when(F.col(column_name) >= top_20_score, F.lit('E')).otherwise(F.col('five_point_class')) )
valid_citations = valid_citations.withColumn('five_point_class', F.when(F.col(column_name) >= top_10_score, F.lit('D')).otherwise(F.col('five_point_class')) )
valid_citations = valid_citations.withColumn('five_point_class', F.when(F.col(column_name) >= top_1_score, F.lit('C')).otherwise(F.col('five_point_class')) )
valid_citations = valid_citations.withColumn('five_point_class', F.when(F.col(column_name) >= top_01_score, F.lit('B')).otherwise(F.col('five_point_class')) )
valid_citations = valid_citations.withColumn('five_point_class', F.when(F.col(column_name) >= top_001_score, F.lit('A')).otherwise(F.col('five_point_class')) )


print ("Finished! Writing output to file...")

########################################################################################################################################################
## WRITE THE OUTPUT OF CC
if limit_year:
	prefix = str(limit_year) + '-year_' + prefix
# Write output to file
if mode == 'local':
	input_file_name = input_file.split('/')[-1]
	valid_citations.write.format('csv').mode('overwrite').option('delimiter',"\t").option('header',True).save(prefix + input_file_name)
	
elif mode == 'distributed':
	input_file_name = input_file.split('/')[-1]
	# If we read from a folder, write output in another folder, same directory
	if not input_file_name:
		input_file_name = input_file.split('/')[-2]
		output_prefix = '/'.join(input_file.split('/')[:-2])
	else:
		output_prefix = '/'.join(input_file.split('/')[:-1])
		
	print ("Writing output at: ")
	print (output_prefix + "/" + prefix + input_file_name)
	valid_citations.write.mode('overwrite').option('delimiter','\t').option('header',True).csv(output_prefix + '/' + prefix + input_file_name)
########################################################################################################################################################

print ("\n\nDone!\n\n")
###################
# Terminate Spark #
sc.stop()	  #
sys.exit(0)	  #
###################

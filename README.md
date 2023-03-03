# Bip-Ranker
Spark-based Paper Ranking Scripts used in Bip! Finder and openAIRE

This is a collection of ranking scripts written in PySpark. 
The collection included in this repository is tailored to run on the openaire cluster, using its gateway node for running them.

The scripts were written in python3, however, the openaire cluster uses python2.7. The only addition that was required for the scripts to be 
compatible, was to import the print function from the future package. To run the scripts on a python3-cluster, the corresponding import lines 
may need to be commented out.

When invoking any of the scripts from the command line without any arguments provided, they will output the expected input parameters and exit.
For a quick refresher on what to pass in the command line, first try this, before delving into the details provided in the following sections.

## Openaire gateway - cluster setup

On the openaire gateway to the cluster, all distributed computation infrastructures have been set up in a particular configuration.
In this setup, running 

> hdfs dfs -ls 

without any other argument will display the contents of hadoop's hdfs under /user/<user_name>/.
When running any command on hdfs, by default, it will consider any relative path as being included under this directory.

However, the scripts expect a full input path for all files, in order to work. Hence, if we have graph_file.txt.gz under
/user/<user_name>, we will need to provide it from the command line as: 

> hdfs:///user/<user_name>/graph_file.txt.gz

**ΝΟΤΕ:** spark is faster when it loads data from a directory containing partitioned files. Hence, it is advised to have the graph split in many 
smaller parts (e.g., equan to the suggested number of partitions in the cluster - 7680 on openaire's IIS) and have it in a golder e.g., 
/user/<user_name>/path/to/graph_input_folder/



## Graph input

The graph input files expected by all ranking scripts are all expected to be in KanellosTM format. Each line of this format consists of the following
tab-separated columns:

> <paper_id> <tab> < <comma_separated_list_of_referenced_papers | <num_referenced_papers> | initial_paper_score > <tab> <previous_score> <tab> <publication_year>
  
**NOTE:** Since we provide a comma separated list of referenced papers, no paper id should include the character ",". 
  So far we have dealt with this by replacing "," with the string "comma_char", then running the ranking, and after finishing, replacing
  "comma_char", again with ",". The same holds for the tab character, which is used to separate the input columns.
  
**NOTE2:** When a paper has no references, then the comma separated list of reference papers is the string "0" and the same holds for the number of referenced_papers.
  
## Notes on Iterative Methods
  
Due to some of the methods being iterative, some spark-computation specific parameters are required. Specifically, since particular dataframes are
computed iteratively, we need to **checkpoint** them at each iteration. Checkpointing materializes a dataframe and truncates its computation history
(which in iterative algorithms can lead to out of memory errors). The materialization can be done on the hdfs, or be performed in the local memory of 
the various nodes involved in the computations. Our implementation provides the option to choose one of these two modes throught he parameter 
checkpoint_mode.  For very large graphs (e.g., Bip, openaire), materialization should be done on the hdfs, since data volatility will lead to the 
scripts failing. On smaller graphs (e.g., dblp) this can be done using local checkpointing. Naturally, checkpointing on hdfs is IO-heavy, and hence 
leads to slower execution, however our trials so far show that for large graphs (e.g., Bip), using local checkpointing leads to errors and failure.

## Output format of Scripts

All scripts of this repository produce their output in a common format (detailed at the end of this Section).
  
Apart from the ranking score calculation itself, which differs per script, all scripts also include a common post-processing procedure, 
which adds to the output files information about the class of each paper based on its score (i.e., the percentile of top scores which they
fall in).. Additionally this post processing step writes tou the standard output the cutoff scores separating the various percentiles. These
messages are required to facilitate the updates in FiP! Finder's data).
  
Te output format of all scripts is a tab separated 3-column format consisting of the follofinw data: 
  
 > <paper_id> <ranking_score> <paper_class>
  
 he paper class takes the values {C1, C2, C3, C4, C5}, which correspond to papers belonging to the top {0.01, 0.1, 1, 10} percentiles for classes 
 C1-C4, respectively, and the botton 90% for class C5.

  
## PageRank.py
  
This script implements google's pagerank method on a citation graph. The expected pagerank-specific parameter is the probability a, which determines
probabilities of following references/links vs performing random jumps. The method is iterative and also takes spark-computation specific parameters.
The resulting file is named after the graph input, prefixed by "PR_" and including information on the alpha parameter and number of iterations. It will
be found in the same directory as the input graph.

Run the script as follows (7680 corresponds to the number of partitions, which we use based on other scripts that run on openaire cluster):
  
> spark2-submit --executor-memory 7G --executor-cores 4 --driver-memory 7G PageRank.py  hdfs:///user/<user_name>/<path_to_input_graph> <alpha> <convergence_error> <checkpoint_directory> 7680 <checkpoint_mode: dfs or local>
  
  
## CC.py
  
This script implements both the citation count, as well as calculating the impulse (i.e., the citation count up to n years after publication) of a paper.
It takes as input the graph file, the number of partitions used, as well as an optional argument determining the n years to use in order to calculate the impulse. Result files will be found under the same directory in hdfs as the graph input file, and their name it the graph file name prefixed by "CC_" or "N-year_CC_", depending on the mode used to run it. 
  
When running from command line, some spark specific parameters are provided. In the following examples we present the values used on the openaire cluster. Further, based on examples of scripts run on the openaire cluster by other members, the number of partitions used is 7680.
  
For vanilla CC run as:
  
>  spark2-submit --executor-memory 7G --executor-cores 4 --driver-memory 7G CC.py hdfs:///user/<user_name>/<path_to_graph_file_on_hdfs> 7680
 
To use impulse mode, run as:
  
>   spark2-submit --executor-memory 7G --executor-cores 4 --driver-memory 7G CC.py hdfs:///user/<user_name>/<path_to_graph_file_on_hdfs> 7680 <num_of_impulse_years>
  
## RAM.py
  
This script implements the RAM / ECM methods described in the paper "Time-aware ranking in dynamic citation networks". Essentially, RAM is a modified 
citation count, where each citation is considered more important the more recently it was made. ECM is a variation that counts weighted citation paths
that lead to each paper, where each citation weight is discounted based on when it was made, and how far in the citation path it occurs. The result
file is written in the same folder as the input graph in hdfs, prefixed by RAM_ or ECM_ depending on the mode. Additionally, the filename includes 
information on the parameters used to run the script (exponential base gamma for simple RAM and gamma and attenuation factor alpha for ECM). Additionally
the scripts require some execution mode specific parameters, such as the number of partitions, the checkpointing directory and the checkpointing mode.
  
Run the script as:
  
>  spark2-submit  --executor-memory 7G --executor-cores 4 --driver-memory 7G TAR.py hdfs:///user/<user_name>/<path_to_graph_file_on_hdfs> <gamma> <current_year> <mode:RAM or ECM> 7680 <checkpoint_directory> <checkpoint_mode: dfs or local> <optional: alpha (only for ECM mode)> <optional: convergence error (only for ECM mode)>
  
7680 corresponds to the number of partitions we use on the openaire cluster.
  
## AttRank.py
  
This script implements the AttRank method (Attention-based Ranking), proposed by Kanellos et al. It is a variation of PageRank where instead of 
Random Jumps, the random surfer chooses papers with preference to those recently published or recently cited. The script takes as parameters the weights
of the probabilities of following references (alpha), the probability of choosing a recently cited paper (beta) the probability of choosing a paper 
based on its age (gamma), the exponential factor based on which age-based probabilities are calculated (exponential_rho). Further it takes as input
the current year and the year from which onward we calculate its recent attention (<start_year_for_attention>). The convergence error determines when
to stop iterations. Finally, the checkpoint directory sets where checkpoints of datafames are saved, 7680 corresponds to the number of partitions, 
and the checkpoint mode determines whether local checkpoints of dataframes are made, or if they are written to disk.
  
Run the script as:
  
> spark2-submit --executor-memory 7G --executor-cores 4 --driver-memory 7G AttRank.py hdfs:///user/<user_name>/<path_to_graph_file_on_hdfs> <alpha> <beta> <gamma> <exponential_rho> <current_year> <start_year_for_attention> <convergence_error> <ceckpoint_directory> 7680 <checkpoint_mode: dfs or local>
  
## Moving scripts to other clusters
  
To run these scripts on another cluster, some lines of code may need to be changed and additional spark-specific parameters may need to be passed in the command line.
 
Particular lines that may require changes:
  * from __future__ import print_function: remove this line if the cluster is python3 compatible
  * master_prefix = "/".join(input_file.split("/")[:5]): where applicable, this line may need to change. It determines the prefix of the output directory, based on how files in the hdfs are referenced (this is setup specific). As is, it works for calling input graphs referenced as hdfs:///user/<user_name>/<path_to_graph_file_on_hdfs>

    
# Please Cite
  
If you use our scripts, for RAM / PageRank / CC, please cite:
 
> Kanellos I, Vergoulis T, Sacharidis D, Dalamagas T, Vassiliou Y. Impact-based ranking of scientific publications: a survey and experimental evaluation. IEEE Transactions on Knowledge and Data Engineering. 2019 Sep 13;33(4):1567-84.
 
> Vergoulis T, Chatzopoulos S, Kanellos I, Deligiannis P, Tryfonopoulos C, Dalamagas T. Bip! finder: Facilitating scientific literature search by exploiting impact-based ranking. InProceedings of the 28th ACM International Conference on Information and Knowledge Management 2019 Nov 3 (pp. 2937-2940).
  
> Vergoulis T, Kanellos I, Atzori C, Mannocci A, Chatzopoulos S, Bruzzo SL, Manola N, Manghi P. Bip! db: A dataset of impact measures for scientific publications. InCompanion Proceedings of the Web Conference 2021 2021 Apr 19 (pp. 456-460).

(of course you should also cite the original works based on which these methods were implemented)
  
For AttRank please cite:
 
> Kanellos I, Vergoulis T, Sacharidis D, Dalamagas T, Vassiliou Y. Ranking papers by their short-term scientific impact. In2021 IEEE 37th International Conference on Data Engineering (ICDE) 2021 Apr 19 (pp. 1997-2002). IEEE.


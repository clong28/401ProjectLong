TensorFlow is an open-source software library developed by Google for numerical computation and large-scale machine learning. It is primarily a programming tool that allows users to build and train models for tasks like classification, regression, and clustering. In science and engineering, TensorFlow is widely used for data analysis, simulations, and predictive modeling. Forecasting experimental outcomes, identifying patterns in large datasets, or optimizing engineering systems. Its flexibility and scalability make it valuable in both research and applied settings, from developing AI-powered diagnostics in healthcare to modeling physical systems in engineering.

For this experimental portion there are a couple libraries that require installation on the HPCC and their quick description is below. 
* Tensorflow - An open-source framework developed by Google for machine learning and deep learning tasks. It enables users to build, train, and deploy models efficiently across CPUs, GPUs, and TPUs.
* Python/3.12.3-GCCcore-13.3.0
* NumPy - Python library for numerical computing, providing support for large, multi-dimensional arrays and matrices. It offers efficient array operations and linear algebra routines widely used in scientific and engineering applications.
* Pandas - Pandas is a powerful data analysis library that offers data structures like DataFrames for handling structured data. It simplifies data cleaning, manipulation, and analysis with intuitive, high-level syntax.
* CUDA - CUDA is a parallel computing platform by NVIDIA that allows developers to use GPUs for general-purpose processing. It dramatically speeds up compute-intensive tasks in areas like deep learning, simulations, and image processing.
* Scikit-learn - Machine learning library built on top of NumPy and SciPy, offering tools for classification, regression, clustering, and model evaluation. It’s widely used for building and benchmarking models in both academic and industry projects.
---
All of these libraries are installed within the submission script as installing Tensorflow will create directories that you may not want on your local storage.
Within this folder will contain the submission script that runs a model toidentify key features that lead to a NCAA team being chosen to compete in the NCAA tournament.
After running the submission script the results of the model will be in the file named ```results.out```.

---
### Installing Packages
* All installation required will be within the submission script. To view what is being installed run the command ```cat submission_script.sb```.
---
### To Submit the Submission Script follow these instructions
* Make sure all files within this folder (.tensor_flow-info/) are in the same folder. So your folder should contain: ```example.py, cbb.csv, and submission_script.sb```.
* Use the following command: ```sbatch submission_script.sb``` to submit the job to the HPCC.
---
## To view the results
* To monitor whether the job is still running use this command: ```squeue -u $USER```. To find your $USER look on the command line that shows where you are in the directory and it is the first thing before the dev node you are on.
* So for example, to view if my job was still running I would check what my $USER is, longcon1@dev-amd20-v100 is the dev node I am working on so my $USER is *longcon1*. So ```squeue -u longcon1``` will display if the job is running. 
* Once the job is completed check if results.out is within your directory by doing ```ls```. To view the results do the command ```cat results.out``` to view the model's most important features. Ignore warnings as they are meant for optimization, which I have not completed.

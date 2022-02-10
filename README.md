# Task solution

The solution for the task is in the file 'script.py' on the master branch.
The series and country code are passed to program through the command line interface, for example, for the series code 'NY.GDP.MKTP.CN' and the country code 'afg' one should run the following command

    python script.py --series_code=NY.GDP.MKTP.CN --country_code=afg

Output is saved in the file 'output.json'.

For completeness program includes exception handling and logging. Logging is saved in the file 'execution_log.log'
Additionally, models are put in the wrapper classes to enable polymorphic behaviour for the ease of extensibility by additional models. 


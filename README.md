# Task solution

The solution for the task is in the file named scirpt.py.
The series and country code are passed to program through command line interface, for example, for series code 'NY.GDP.MKTP.CN' and the country code 'afg' one should run command

*python script.py --series_code=NY.GDP.MKTP.CN --countr_code=afg*

For completness program includes exception handling and logging. 
Additionally models are put int wrapper classes to enable polymorphic behaviour, for the ease of extensibility by additional models. 


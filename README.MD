# Wersje

<b>v0.0.1</b>

Jupyter notebook:

<ul>
    <li> generating data
    <li> get features
</ul>

<b>v0.0.2</b>

<ul>
    <li> Functions moved from jupyter notebook to normal python files.
    <li> Added multithreating in get_features function
</ul>

<b>v0.0.3</b>

<ul>
    <li> Added ability to limit readed trajectories
    <li> Repaired lack of T in gaussinity
    <li> Repaired fractal dimension measurment
    <li> Added function that repair old data generated with wrong fractal dimmension
</ul>

<b>v0.0.4</b>

<ul>
    <li> Added estimation of D and &alpha; by tMSD
    <li> Moved TAMSD function from TAMSD.py to get_features.py
</ul>

<b>v0.1.0</b>

<ul>
    <li> Global
    <ul>
        <li> Jupyter notebook removed
        <li> test removed
        <li> Added ,,readme''
        <li> Added different modes in functions depending on part of experiment
        <li> Added interface to comunicate with user
        <li> Removed unused parameters from <i>,,global_params.py''</i>
    </ul>
    <li> Generating trajectories
    <ul>
        <li> New way to generate trajectories implemented
        <li> Added generated trajectories
    </ul>
    <li> TAMSD
    <ul>
        <li> Fully rebuilded and repaired TAMSD
        <li> Added ability to write and read estimations
        <li> Added estimated parameters and tamsd points from trajectories
    </ul>
    <li> Plotting
    <ul>
        <li> Added plotting of trajectories
        <li> Added plotting of TAMSD
        <li> Added plots of trajectories
        <li> Added plots of TAMSD
    </ul>
    <li> [<font color='red'>ERROR</font>] To repair get_features (rework for new data mode)
</ul>

<b>v0.1.1</b>

<ul>
    <li> Global
    <ul>
        <li> TAMSD moved from TAMSD.py to get_features.py
        <li> Added functions to split and join too big for github pandas dataframes
    </ul>
    <li> Generating and reading trajectories
    <ul>
        <li> Added missing logs
    </ul>
    <li> Get features
    <ul>
        <li> Get_features reworked to new data generated
        <li> Removed function repairing fractal dimmension (no longer usefull)
        <li> Added features from trajectories
    </ul>
</ul>

<b>v0.1.2</b>

<ul>
    <li> Changed github to divide all file types (not only )
    <li> Removed printing of dataframe after join
    <li> Generating and reading trajectories
    <ul>
        <li> Added function reading features for Machine Learning
        <li> Saved models and estimations for new methods
    </ul>
    <li> Get features
    <ul>
        <li> Changed listed values to first and five
    </ul>
    <li> Machine Learning
    <ul>
        <li> Added multidimension linear regression method.
        <li> Added decision tree method.
        <li> Added decision tree plot after generate
        <li> Added random forest method.
    </ul>
    <li> Testing
    <ul>
        <li> Added R^2 test.
        <li> Added tests based on eps.
        <li> Added max_error.
    </ul>
</ul>
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
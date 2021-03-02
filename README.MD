# Determination of the anomalous exponent via machine learning methods
## Info
Programs that were used while writing my Engineering Thesis.

The main task is to check some feature-based machine learning alghorithms are better than TAMSD (classic method) in estimation of anomalous diffusion coefficient.

Features extracted from trejectories are:
<ul>
    <li> współczynnik dyfuzji $D$, wydajność $E$, współczynnik spowolnienia $s$, iloraz MSD $\kappa$, współczynnik anty-gaussowości $G$, liniowość $S$, autokorelacja $R$, </li>
    <li> maksymalna odległość między punktami $d_\max$, prawdopodobieństwo uwięzienia $P$, wymiar fraktalny $D_f$. </li>
</ul>

Checked alghoritms are:
<ul>
    <li> multivariable linear regression </li>
    <li> decision tree </li>
    <li> random forest </li>
    <li> gradient boosting </li>
</ul>

Data is generated using AnDi-dataset [https://github.com/AnDiChallenge/ANDI_datasets].

For more info check ,,Praca inżynierska.pdf''.

## Usage
<b>To generate trajectories, get features from them and learn models</b> run ,,main.py'' file.
As part can be choosen numbers from 1 to 10. Each generates different data:
<ol>
    <li> 110000 trajectories with random noise from Brownian Motion only. </li>
    <li> 110000 trajectories with random noise from all awailable motions. </li>
    <li> 110000 trajectories with noise ration equal 0 from all awailable motions. </li>
    <li> 110000 trajectories with noise ration equal 0.1 from all awailable motions. </li>
    <li> 110000 trajectories with noise ration equal 0.3 from all awailable motions. </li>
    <li> 110000 trajectories with noise ration equal 1 from all awailable motions. </li>
    <li> 110000 trajectories with random noise from all awailable motions. </li>
    <li> 11 trajectories with random noise from all awailable motions. </li>
    <li> 110 trajectories with random noise from all awailable motions. </li>
    <li> 1100 trajectories with random noise from all awailable motions. </li>
    <li> 11000 trajectories with random noise from all awailable motions. </li>
</ol>

Then answear to program a questions that it will ask.

<b>To test models, get prediction values and get prediction quality</b> use ,,test.py'' file, but it's draft version, so you need to uncomment certain parts and sometimes change code to get what you want.

## Changelog

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

<b>v0.1.3</b>

<ul>
    <li> Changed project layout for 3 model types.
    <li> Added log file and time measurment
    <li> Changed plotting format from png to pdf
    <li> Git now don't get datasets. Only programs.
    <li> Machine Learning
    <ul>
        <li> Added gradient boosting regressor.
    </ul>
    <li> Part 0
    <ul>
        <li> Added plotting of example trajectories
        <li> Added plotting of example and perfect TAMSD
    </ul>
    <li> Get features
    <ul>
        <li> Repaired badly counted TAMSD.
        <li> Repaired badly counted TAMSD4.
        <li> Repaired badly counted straightness.
    </ul>
    <li> Machine Learning
    <ul>
        <li> Added some safe options to decision tree and random forest regressors.
    </ul>
</ul>

<b>v0.1.4</b>

<ul>
    <li> Added some testing options
    <li> Changed TAMSD estimation
</ul>

<b>v0.2.0</b>

<ul>
    <li> Added second part of everything (only fractal Brownian motion)
    <li> Added modified andi_dataset function
    <li> Changed cla() to close() for plots in test module
    <li> Saved logs in separated folders for every part
    <li> Changed learning rates for gradient boosting
</ul>

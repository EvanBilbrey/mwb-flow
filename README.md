# mwb-flow
 Monthly Water Balance Runoff Model

 Uses the following conda commands to set up env
conda create -n mwb_flow python=3.9
conda activate mwb_flow
pip install git+https://github.com/MTDNRC-WRD/GRIDtools@main
pip install rasterstats
pip install git+https://github.com/MTDNRC-WRD/chmdata@main
conda install -c conda-forge py3dep
conda install ipykernel tqdm tomli
pip install pynhd
pip install matplotlib

pip install spotpy
@echo off
echo Creating virtual environment...
python -m venv venv --without-pip

echo Downloading get-pip.py...
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

echo Installing pip...
.\venv\Scripts\python get-pip.py

echo Activating virtual environment...
call .\venv\Scripts\activate

echo Installing requirements...
pip install -r requirements.txt

echo Creating data directory...
mkdir data

echo Setup complete!
echo To activate the virtual environment, use:
echo .\venv\Scripts\activate

echo Cleaning up...
del get-pip.py

pause 
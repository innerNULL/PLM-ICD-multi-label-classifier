# file: etl_fetch_mimic3_demo_dataset.sh
# date: 2023-09-22
#
# For data details, pls refer to https://physionet.org/content/mimiciii-demo/1.4/


set -x


CURR_DIR=$(pwd)
DATA_ROOT_DIR=$1


function init() {
  mkdir -p ${DATA_ROOT_DIR}/raw_data
}


function download() {
  cd ${DATA_ROOT_DIR}/raw_data
  wget https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip
  unzip *.zip
  rm *.zip
  cd ${CURR_DIR}
}


function main() {
  init
  download
}


main

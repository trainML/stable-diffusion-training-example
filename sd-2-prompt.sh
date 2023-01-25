#!/bin/bash

for i in "$@"; do
  case $i in
    -s=*|--samples=*)
      SAMPLES="${i#*=}"
      shift 
      ;;
    -i=*|--iters=*)
      ITERS="${i#*=}"
      shift 
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

cd stable-diffusion-2
pip install -r requirements.txt

python scripts/txt2img.py \
--ckpt ${TRAINML_CHECKPOINT_PATH}/v2-1_768-ema-pruned.ckpt \
--config ${TRAINML_CHECKPOINT_PATH}/v2-1_768-ema-pruned.yaml \
--H 768 --W 768 \
--n_iter ${ITERS} --n_samples ${SAMPLES} \
--prompt "${1}"

## images are in the samples subfolder of the script's default output path
cp outputs/txt2img-samples/samples/* $TRAINML_OUTPUT_PATH/
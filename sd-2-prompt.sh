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

python inference.py \
--H 768 --W 768 \
--n_iter ${ITERS} --n_samples ${SAMPLES} \
--prompt "${1}"

#!/bin/bash

for i in "$@"; do
  case $i in
    -s=*|--steps=*)
      STEPS="${i#*=}"
      shift 
      ;;
    -i=*|--images=*)
      IMAGES="${i#*=}"
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


#cp -r ${TRAINML_DATA_PATH}/* . 

cd diffusers/examples/dreambooth

## Disable accelerate logging and increase verbosity
sed -i '/import os/aimport sys' train_dreambooth.py
# sed -i '/import os/aimport logging' train_dreambooth.py
sed -i 's/logger = get_logger(__name__)/logger = logging.getLogger()/' train_dreambooth.py
sed -i 's/logger\./logging./' train_dreambooth.py
sed -i '/logger = logging.getLogger()/alogger.setLevel(logging.INFO)' train_dreambooth.py
sed -i '/logger = logging.getLogger()/alogger.addHandler(logging.StreamHandler(sys.stderr))' train_dreambooth.py
sed -i 's/if accelerator.is_local_main_process:/if True:/' train_dreambooth.py
sed -i 's/disable=not accelerator.is_local_main_process/disable=False/' train_dreambooth.py
sed -i 's/accelerator.state, main_process_only=False/accelerator.state/' train_dreambooth.py

## Run training
python train_dreambooth.py \
--pretrained_model_name_or_path=${TRAINML_CHECKPOINT_PATH} \
--instance_data_dir=${TRAINML_DATA_PATH}/instance-data \
--class_data_dir=${TRAINML_DATA_PATH}/regularization-data \
--output_dir=${TRAINML_OUTPUT_PATH}  \
--with_prior_preservation --prior_loss_weight=1.0 \
--instance_prompt="${2}" \
--class_prompt="${1}" \
--resolution=768  \
--train_batch_size=1  \
--sample_batch_size=1 \
--gradient_accumulation_steps=2 --gradient_checkpointing  \
--use_8bit_adam  \
--learning_rate=5e-6  \
--lr_scheduler="constant"  \
--lr_warmup_steps=0  \
--num_class_images=${IMAGES} \
--max_train_steps=${STEPS} \
--checkpointing_steps=$((STEPS+1)) \
--mixed_precision=bf16 \
--enable_xformers_memory_efficient_attention \
--prior_generation_precision=bf16 \
--allow_tf32
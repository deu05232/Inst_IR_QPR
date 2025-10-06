import logging
import os
import sys
import torch

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

from tevatron.retriever.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments
    
from dataset import (FixedOriginalTrainDataset, FixedTrainDataset, InstructTrainDataset, SplitedFixedTrainDataset,
                     Idea3TrainDataset, Idea3InstructTrainDataset, Idea3SplitedTrainDataset) 

from collator import (TrainCollator, Idea3TrainCollator)

from modeling import (DenseModel, DenseModel_Idea3, DenseModel_Idea3_MSE, DenseModel_Idea3_reduced_KL, DenseModel_Idea3_MSE_rep,
                      DenseModel_Idea3_reduced_KL_only_inst, DenseModel_Idea3_MSE_rep_only_inst)


from trainer import (TevatronTrainer,
                     Idea3TevatronTrainer,
                     Idea3TevatronTrainer_only_inst)


from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    print(f"Tokenizer name is {model_args.tokenizer_name}")
    # print(tokenizer)
        
    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
            
        
    if "no_weight" in training_args.output_dir:  # Original (=Promptriever)
        print("***************Using Dense Model***************")
        model = DenseModel.build(
            model_args,
            training_args,
            cache_dir=model_args.cache_dir,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )  

        if "split" in training_args.output_dir:
            train_dataset = SplitedFixedTrainDataset(data_args)
        elif "-Instruct" in model_args.model_name_or_path:
            print("***************Using Instruct Model***************")
            train_dataset = InstructTrainDataset(data_args, tokenizer=tokenizer, model_name=model_args.model_name_or_path.lower())
        elif "original_data" in training_args.output_dir:
            print("***************Using original data Model***************")
            train_dataset = FixedOriginalTrainDataset(data_args)
        else:
            train_dataset = FixedTrainDataset(data_args)
        collator = TrainCollator(data_args, tokenizer)
        trainer_cls = TevatronTrainer
     
    elif "idea3" in training_args.output_dir:
        if "MSE_rep" in training_args.output_dir:
            if "repllama_init" in training_args.output_dir:
                print("***************Using Dense Model with Idea3 (MSE_rep-Repllama init)***************")
                model = DenseModel_Idea3_MSE_rep_only_inst.build(
                    model_args,
                    training_args,
                    cache_dir=model_args.cache_dir,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                )
                train_dataset = Idea3TrainDataset(data_args)
                collator = Idea3TrainCollator(data_args, tokenizer)
                trainer_cls = Idea3TevatronTrainer_only_inst
            else:
                model = DenseModel_Idea3_MSE_rep.build(
                    model_args,
                    training_args,
                    cache_dir=model_args.cache_dir,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                )  
                if "Instruct" in training_args.output_dir:
                    print("***************Using Dense Model with Idea3 (MSE_rep-Instruct)***************")
                    train_dataset = Idea3InstructTrainDataset(data_args, tokenizer=tokenizer, model_name=model_args.model_name_or_path.lower())
                else:
                    print("***************Using Dense Model with Idea3 (MSE_rep)***************")
                    train_dataset = Idea3TrainDataset(data_args)
                collator = Idea3TrainCollator(data_args, tokenizer)
                trainer_cls = Idea3TevatronTrainer

        elif "MSE" in training_args.output_dir:
            model = DenseModel_Idea3_MSE.build(
                model_args,
                training_args,
                cache_dir=model_args.cache_dir,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )  

            if "hard" in training_args.output_dir and "no_hard" not in training_args.output_dir:
                model_use_idx = 1
            else:
                model_use_idx = 5

            model.kl_temp = 0.01
            model.kl_loss_weight = 0.1
            model.use_idx = model_use_idx

            if "MSE-Instruct" in training_args.output_dir:
                print(f"***************Using Dense Model with Idea3 (MSE-Instruct, t:{model.kl_temp}, l:{model.kl_loss_weight}, idx:{model.use_idx})***************")
                train_dataset = Idea3InstructTrainDataset(data_args, tokenizer=tokenizer, model_name=model_args.model_name_or_path.lower())
            elif "split" in training_args.output_dir:
                print("***************Using Dense Model with Idea3 (MSE-Split)***************")
                train_dataset = Idea3SplitedTrainDataset(data_args)
            else:  # original
                print(f"***************Using Dense Model with Idea3 (MSE, t:{model.kl_temp}, l:{model.kl_loss_weight}, idx:{model.use_idx})***************")
                train_dataset = Idea3TrainDataset(data_args)
            collator = Idea3TrainCollator(data_args, tokenizer)
            trainer_cls = Idea3TevatronTrainer
            
        elif "reduced_KL" in training_args.output_dir:
            if "repllama_init" in training_args.output_dir:
                print("***************Using Idea3 (reduced_KL - repllama init)***************")
                model = DenseModel_Idea3_reduced_KL_only_inst.build(
                    model_args,
                    training_args,
                    cache_dir=model_args.cache_dir,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                )

                train_dataset = Idea3TrainDataset(data_args)
                collator = Idea3TrainCollator(data_args, tokenizer)
                trainer_cls = Idea3TevatronTrainer_only_inst

            else:
                print("***************Using Idea3 (reduced_KL)***************")
                model = DenseModel_Idea3_reduced_KL.build(
                    model_args,
                    training_args,
                    cache_dir=model_args.cache_dir,
                    torch_dtype=torch_dtype,
                    attn_implementation=model_args.attn_implementation,
                )  

                train_dataset = Idea3TrainDataset(data_args)
                collator = Idea3TrainCollator(data_args, tokenizer)
                trainer_cls = Idea3TevatronTrainer
        else:
            print("***************Using Idea3***************")  # KL divergence
            model = DenseModel_Idea3.build(
                model_args,
                training_args,
                cache_dir=model_args.cache_dir,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )  

            train_dataset = Idea3TrainDataset(data_args)
            collator = Idea3TrainCollator(data_args, tokenizer)
            trainer_cls = Idea3TevatronTrainer
    
    else:
        raise NotImplementedError("Check your output_dir")
        

    # trainer_cls = GCTrainer if training_args.grad_cache else TevatronTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        dont_shuffle=model_args.dont_shuffle
    )
    train_dataset.trainer = trainer

    if training_args.resume_from_checkpoint is not None:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

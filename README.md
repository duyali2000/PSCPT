# Beyond the Lexical Consistency: Leveraging Program Semantics for Program Translation
![backbone](backbone.png)
This repo has the PyTorch implementation and datasets of our ASE 2023 paper: Beyond the Lexical Consistency: Leveraging Program Semantics for Program Translation

## Introduction
Program Translation aims to convert the input programs from one programming language to another. Automatic program translation is a prized target of software engineering research, which leverages the reusability of projects and improves the efficiency of development. Recently, thanks to the rapid development of deep learning model architectures and the availability of large-scale parallel corpus of programs, the performance of program translation has been greatly improved. However, the existing program translation models are still far from satisfactory, in terms of the quality of generated programs. In this paper, we argue that a major limitation of the current approaches is that they only consider lexical consistency as the training objective, but misses the semantic consistency between the generated program and the target program, which is also critical for the task. To make the program translation model more semantically aware, we propose a general framework named \textbf{P}reserving \textbf{S}emantic \textbf{C}onsistency for \textbf{P}rogram \textbf{T}ranslation (\textbf{PSCPT}), which considers both lexical and semantic consistency in the training process of program translation and can be easily applied to all encoder-decoder methods with various neural networks (\textit{e.g,} LSTM, Transformer) as the backbone.
We conduct extensive experiments on the benchmark dataset \textit{CoST}, which consists of both snippet-level and program-level parallel data from 7 programming languages and up to 42 programming language pairs. 
Experimental results show that with CodeBERT as the backbone, our approach outperforms not only the state-of-the-art open-source models but also the commercial closed large language model (\textit{e.g.,} CodeX) on the program translation task.


## Requirements
* Conda
  * install conda: [https://conda.io/projects/conda/en/latest/user-guide/install/index.html](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
  * create a new conda environment:
      * if you are running with GPU: 
        ```
        conda env create -f environment-gpu.yml
        conda activate mqmc
        ```
        Dependencies include support for CUDA_11.4. If you are using a different CUDA version update the dependencies accordingly.
      * if you are running with CPU:   
        ```
        conda env create -f environment-cpu.yml
        conda activate mqmc
        ```

## Example to Run the Codes 
The instruction of commands has been clearly stated in the codes.

* Java-C#
```
python run.py --model_type='roberta' --output_dir='saved_models/Java-C#/' --train_filename='../data/program_data/Java-C#/train-Java-C#-tok.cs,../data/program_data/Java-C#/train-Java-C#-tok.java,../data/map_data/C#-program-tok.jsonl,../data/program_data/Java-C#/train-C#-map.jsonl@' --test_filename='../data/program_data/Java-C#/test-Java-C#-tok.cs,../data/program_data/Java-C#/test-Java-C#-tok.java,../data/map_data/C#-program-tok.jsonl,../data/program_data/Java-C#/test-C#-map.jsonl' --train_snippet_filename=None --dev_snippet_filename=None --test_snippet_filename=None --source_lang='Java' --target_lang='C#' --max_source_length=512 --max_target_length=512 --max_comment_length=64 --do_train --do_eval --do_test --train_batch_size=128 --eval_batch_size=128 --learning_rate=0.0001 --beam_size=10 --weight_decay=0.0 --adam_epsilon=1e-08 --max_grad_norm=1.0 --num_train_epochs=200 --max_steps=-1 --eval_steps=-1 --train_steps=-1 --warmup_steps=0 --local_rank=-1 --seed=42 --temperature=2.0 --weightAB=1.0 --weightBB=2.0 --weightcon=3.0 --autonum=50
```

## Contact
If You find any problems or have any questions, please contact with me.


## Acknowledgments and Licenses
* The calculation of CodeBLEU in our work is adapted from CodeBLEU (https://github.com/microsoft/CodeXGLUE/).
* The public CosT dataset used in our work is from MuST-CoST (https://github.com/reddy-lab-code-research/MuST-CoST/)
* All license clauses are in the LICENSE file.

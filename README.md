# ESCoT: Towards Interpretable Emotional Support Dialogue Systems

<img src="https://img.shields.io/badge/Venue-ACL--24-278ea5" alt="venue"/> <img src="https://img.shields.io/badge/Status-Accepted-success" alt="status"/>

This is the repository of our ACL 2024 main paper "[**ESCoT: Towards Interpretable Emotional Support Dialogue Systems**](https://aclanthology.org/2024.acl-long.723/)".

## ESD-CoT Dataset

Our ESD-CoT dataset is organized under the `data` folder and is split into three JSON files: `train`, `val`, and `test`. Each file contains samples structured as follows:

```json
{
    "id": ,
    "original_data": {
        "dialog": [
            {
                "speaker": "seeker",
                "content": "Hi, I'm having a really hard time managing my schoolwork and extracurricular activities. I feel like there's just not enough hours in the day."
            },
            ...
            {
                "speaker": "seeker",
                "content": "Yeah, I can try that."
            }
        ],
        "strategy": "Providing Suggestions",
        "response": "Great, and let's touch base next week to see if the list has been helpful. In the meantime, have you considered talking to your teacher or a guidance counselor about feeling overwhelmed?"
    },
    "cot_data": {
        "emotion": "The seeker feels overwhelmed and stretched thin.",
        "emotion_stimuli": "The seeker is struggling to manage schoolwork...",
        "individual_appraisal": "The seeker thinks they are not able to do anything well...",
        "recognized_strategy": "Providing Suggestions",
        "strategy_reason": "To address the seeker's feeling of being overwhelmed and..."
    }
}
```
Additionally, we provide instructional format training data in the `data/ablation_data` folder.

## Model Training

### Download the pretrained models
Download the [**LLAMA2-7B-CHAT**](https://huggingface.co/meta-llama/Llama-2-7b-hf) model.

The training of LLAMA2-CHAT model is based on the [**SFT trainer of Transformer Reinforcement Learning**](https://github.com/huggingface/trl).

### Train Model
Run bash `scripts/supervised_finetune_llama2_cot.sh` to train your model.

Run bash `scripts/supervised_finetune_llama2_cot_ablation.sh` for Ablation Study model training.

### Test Model
Run bash `scripts/test_llama2_chat_sft_cot.sh` or `scripts/test_llama2_inference_cot.sh`.

## Cite
If you use our codes or your research is related to our work, please kindly cite our paper:
```bib
@inproceedings{zhang-etal-2024-escot,
    title = "{ESC}o{T}: Towards Interpretable Emotional Support Dialogue Systems",
    author = "Zhang, Tenggan and Zhang, Xinjie and Zhao, Jinming and Zhou, Li and Jin, Qin",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2024"
}
```

Please contact zhangxinjie827@ruc.edu.cn for any problems.
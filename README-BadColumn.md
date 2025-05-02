Somehow, a bad file got pushed to hugging face. there is a local version that is correct that is here: C:\Users\admin\Stoney-RL\synthetic_stoney_data_fixed3.jsonl

so we need to both push the correct JSON file to replace this dataset: HarleyCooper/synthetic_stoney_data

synthetic_stoney_data_fixed2.jsonl at HF needs to be replaced with this file: 



PS C:\Users\admin\Stoney-RL> docker run --rm --gpus all -v .:/app stoney-rl-local
/opt/conda/lib/python3.10/site-packages/transformers/utils/generic.py:441: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
/opt/conda/lib/python3.10/site-packages/transformers/utils/generic.py:309: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
  _torch_pytree._register_pytree_node(
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
Loading dataset 'HarleyCooper/StoneyNakoda45k' from HuggingFace Hub...
Generating train split: 104 examples [00:00, 1612.81 examples/s]
Traceback (most recent call last):
  File "/opt/conda/lib/python3.10/site-packages/datasets/builder.py", line 1871, in _prepare_split_single
    writer.write_table(table)
  File "/opt/conda/lib/python3.10/site-packages/datasets/arrow_writer.py", line 623, in write_table
    pa_table = table_cast(pa_table, self._schema)
  File "/opt/conda/lib/python3.10/site-packages/datasets/table.py", line 2293, in table_cast
    return cast_table_to_schema(table, schema)
  File "/opt/conda/lib/python3.10/site-packages/datasets/table.py", line 2241, in cast_table_to_schema
    raise CastError(
datasets.table.CastError: Couldn't cast
question: string
answer: string
source_language: string
generated_at: string
pair_id: int64
to
{'messages': [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}]}
because column names don't match

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/train_lora_local.py", line 32, in <module>
    ds = load_dataset(dataset_name, split="train")
  File "/opt/conda/lib/python3.10/site-packages/datasets/load.py", line 2084, in load_dataset
    builder_instance.download_and_prepare(
  File "/opt/conda/lib/python3.10/site-packages/datasets/builder.py", line 925, in download_and_prepare
    self._download_and_prepare(
  File "/opt/conda/lib/python3.10/site-packages/datasets/builder.py", line 1001, in _download_and_prepare
    self._prepare_split(split_generator, **prepare_split_kwargs)
  File "/opt/conda/lib/python3.10/site-packages/datasets/builder.py", line 1742, in _prepare_split
    for job_id, done, content in self._prepare_split_single(
  File "/opt/conda/lib/python3.10/site-packages/datasets/builder.py", line 1873, in _prepare_split_single
    raise DatasetGenerationCastError.from_cast_error(
datasets.exceptions.DatasetGenerationCastError: An error occurred while generating the dataset

All the data files must have the same columns, but at some point there are 5 new columns ({'answer', 'pair_id', 'source_language', 'generated_at', 'question'}) and 1 missing columns ({'messages'}).

This happened while the json dataset builder was generating data using

hf://datasets/HarleyCooper/StoneyNakoda45k/Dictionaries/bilingual_training_set.jsonl (at revision 5798c3407eb5f780d002d63733e6ec5d01b55ac5)        

Please either edit the data files to have matching columns, or separate them into different configurations (see docs at https://hf.co/docs/hub/datasets-manual-configuration#multiple-configurations)
PS C:\Users\admin\Stoney-RL> 
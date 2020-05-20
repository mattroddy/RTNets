import subprocess

subprocess.call("python ./scripts/extract_gemaps.py", shell=True)
subprocess.call("python ./scripts/prepare_gemaps.py", shell=True)
subprocess.call("python ./scripts/extract_mfccs_log_filt_banks.py", shell=True)
subprocess.call("python ./scripts/get_updates_msstate.py", shell=True)
subprocess.call("python ./scripts/get_updates.py", shell=True)
print('Getting glove embeddings...')
print("python ./scripts/create_glove_models.py")
subprocess.call("python ./scripts/create_glove_models.py", shell=True)

import os
import sys

from train_fast import FastTrainer
from config import MODEL_DIR

print('MODEL_DIR from config:', MODEL_DIR)

trainer = FastTrainer()
print('trainer.model_dir:', trainer.model_dir)
print('Path exists:', os.path.exists(trainer.model_dir))

# 测试保存
test_path = trainer._get_model_path('black', 999)
print('Test path:', test_path)

# 尝试保存
try:
    trainer.black_agent.save(test_path)
    print('Save successful!')
    print('File exists:', os.path.exists(test_path))
    if os.path.exists(test_path):
        print('File size:', os.path.getsize(test_path))
        os.remove(test_path)
        print('Test file cleaned up')
except Exception as e:
    print('Save failed:', e)
    import traceback
    traceback.print_exc()

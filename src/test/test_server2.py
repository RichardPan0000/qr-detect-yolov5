from locust import HttpUser, TaskSet, task, between
import os
import random

class PredictTaskSet(TaskSet):
    def on_start(self):
        # 获取所有图片文件的路径
        self.image_folder = '../data/images/test_img2'
        self.image_files = [os.path.join(self.image_folder, f) for f in os.listdir(self.image_folder) if f.endswith('.jpg')]

    @task
    def upload_file(self):
        # 随机选择一个图片文件
        image_path = random.choice(self.image_files)
        with open(image_path, 'rb') as file:
            files = {'file': (os.path.basename(image_path), file, 'image/jpeg')}
            self.client.post("/predict/", files=files)

class PredictUser(HttpUser):
    tasks = [PredictTaskSet]
    wait_time = between(1, 5)
    host = "http://127.0.0.1:5000"

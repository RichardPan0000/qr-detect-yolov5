from locust import HttpUser, TaskSet, task, between

class PredictTaskSet(TaskSet):
    @task
    def upload_file(self):
        # 指定要上传的文件路径
        file_path = "../data/images/test_img/IMG_20241227_153918.jpg    "
        # 打开文件并准备上传
        with open(file_path, "rb") as file:
            # 使用 multipart/form-data 方式上传文件
            files = {"file": (file_path, file, "image/jpeg")}
            # 发送 POST 请求到 /predict/ 端点
            self.client.post("/predict/", files=files)

class PredictUser(HttpUser):
    tasks = [PredictTaskSet]
    # 设置用户等待时间，模拟真实用户行为
    wait_time = between(1, 5)
    # 设置主机地址
    host = "http://127.0.0.1:5000"

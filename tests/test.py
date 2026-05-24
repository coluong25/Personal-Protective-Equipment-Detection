# Get và setter để debug
class MLModel:
    def __init__(self):
        self.__learning_rate = 0.01

    @property
    def learning_rate(self):
        print(f"[GET] learning_rate được đọc → giá trị: {self.__learning_rate}")
        return self.__learning_rate

#     @learning_rate.setter
#     def learning_rate(self, value):
#         print(f"[SET] Đang ghi learning_rate: {self.__learning_rate} → {value}")
#         if value <= 0:
#             raise ValueError("Learning rate phải > 0")
#         self.__learning_rate = value
#         print(f"[SET] Ghi thành công!")

# model = MLModel()

# print(model.learning_rate)
# # [GET] learning_rate được đọc → giá trị: 0.01
# # 0.01

# model.learning_rate = 0.001
# # [SET] Đang ghi learning_rate: 0.01 → 0.001
# # [SET] Ghi thành công!

# model.learning_rate = -1



import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self):
        self.__learning_rate = 0.01
    @property
    def learning_rate(self):
        print(f"[GET] learning_rate được đọc → giá trị: {self.__learning_rate}")
        return self.__learning_rate


    @learning_rate.setter
    def learning_rate(self, value):
        logger.debug(f"Thay đổi learning_rate: {self.__learning_rate} → {value}")
        self.__learning_rate = value

# Output trong terminal:
# DEBUG:__main__:Thay đổi learning_rate: 0.01 → 0.001
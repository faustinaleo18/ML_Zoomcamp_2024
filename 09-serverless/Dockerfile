FROM agrigorev/model-2024-hairstyle:v3
FROM public.ecr.aws/lambda/python:3.10

RUN pip install tensorflow
RUN pip install pillow
RUN pip install numpy==1.23.1
RUN pip install --no-deps https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl

COPY homework.py .
COPY model_2024_hairstyle.tflite .
COPY yf_dokzqy3vcritme8ggnzqlvwa.jpeg .

ENV MODEL_NAME=model_2024_hairstyle.tflite

CMD [ "homework.lambda_handler" ]
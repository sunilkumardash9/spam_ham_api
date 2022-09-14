FROM python:3.8-slim-buster
# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt 

# 
COPY ./model_API.py spam_model spamClassify.py /code/
#
#COPY ./app/spam_model /code/spam_model
#FROM python:3.8-slim-buster
#WORKDIR /app
#COPY --from=bulld /code /app
#
CMD ["uvicorn", "model_API:app", "--host", "0.0.0.0"]



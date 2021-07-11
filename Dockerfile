FROM python:3.9.5
RUN git clone https://github.com/renan-cunha/NameGeneratorBR
RUN cd NameGeneratorBR && make create_environment
RUN cd NameGeneratorBR && make requirements
WORKDIR NameGeneratorBR
ENTRYPOINT ["python", "src/models/predict_model.py"]

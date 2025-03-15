## 설치 방법

### 1. EXE 버전 사용하기
1. [이 링크](
https://drive.google.com/file/d/1m4PPUO05-N1fPiwVjnXsOXIncSqr7Xg6/view?usp=drive_link)를 통해 EXE 파일을 다운로드하세요.
2. 다운로드한 EXE 파일을 실행하여, 모델을 즉시 사용할 수 있습니다.

### 2. Python 환경에서 사용하기
1. Python 3.x와 필요한 라이브러리를 설치하세요.
2. `requirements.txt` 파일을 사용해 의존성 라이브러리를 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```
3. 모델을 로드하고 예측을 시작합니다.
   ```python
   from mindai_model import load_model
   model = load_model('path_to_model.h5')
   predictions = model.predict(input_data)
   ```





## 라이선스

이 프로젝트는 **MIT 라이선스** 하에 배포됩니다. 자세한 내용은 [LICENSE](https://github.com/INSECT5386/MINDAI/blob/main/LICENCE) 파일을 참고하세요.

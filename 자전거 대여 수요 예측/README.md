# ML Project - 자전거 수요 예측
> [프로젝트 kaggle 링크](https://www.kaggle.com/c/bike-sharing-demand)

## 탐색적 데이터 분석
### 데이터 둘러보기
**학습 및 테스트 데이터**

![학습_테스트_데이터](./images/train_test_data.png)

- 테스트 데이터에 casual과 registered 피처가 존재하지 않으므로 이후, 모델을 훈련할 때 해당 피처를 제외한다.

**학습, 테스트 데이터 정보**
<p align="center">
  <img src="./images/train_test_info.png" width="600" height="200"/>
</p>

- DataFrame 각 열의 결측값이 몇 개인지, 데이터 타입은 무엇인지 info()함수를 사용하여 파악한다.

### 피처 엔지니어링
데이터를 다양한 관점에서 시각화해보면 raw data 상태에서는 찾기 어려운 경향, 공통점, 차이 등을 찾을 수 있다. 하지만, datetime과 같은 일부 데이터는 시각화하기에 적합하지 않은 형태일 수도 있기 때문에 변환(피처 엔지니어링)을 해준다.

**datetime -> 데이터 타입: object**
- object 타입은 문자열 타입이라고 볼 수 있다.
- 연도, 월, 일, 시간, 분, 초로 구성되어 있기 때문에 세부적으로 분석하기 위해 구성요소별로 나누어본다.
  - date 피처가 제공하는 정보는 모두 year, month, day 피처에 존재하므로 추후 제거해준다.
- calendar와 datetime 라이브러리를 활용해 요일 피처를 문자로 만들어준다.
- 'season', 'weather' 피처의 경우 범주형 데이터로 이루어져있어 정확한 의미 파악이 힘들다. 시각화를 위해 의미가 잘 드러나도록 map() 함수를 사용하여 문자열로 변환한다.
  - 세 달씩 '월'을 묶으면 '계절'이 되므로, 지나친 세분화를 방지하기 위해 이후, season 피처만 남기고 month 피처는 제거해준다.
<p align="center">
  <img src="./images/FE_data.png">
</p>



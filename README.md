# Mask_R-CNN   
### 1. RCNN 계열 모델 소개   
R-CNN(2014) --------------> for Object Detection   
Fast R-CNN(2015) ---------> for Object Detection   
Faster R-CNN(2016) -------> for Object Detection   
Mask R-CNN(2017) --------> for Instance Segmentation   
   
### 2. R-CNN   
   
![RCNN](https://user-images.githubusercontent.com/59756209/74311226-48da3980-4db2-11ea-99da-f827953f110e.PNG)      
   
Object Detection에 CNN이 사용되기 시작한 것으로 보면, 이미지 분류와 Detection은 밀접한 연관이 있다.   
R-CNN은 처음으로 CNN과 같은 딥러닝 기반의 이미지 분류이다.   
feature extractor로 사용하여 Object Detection, Segmentation에 높은 성능을 보였다.   
   
#### (1) Region proposal algorithm   
대상의 class를 구분하지 않고 이미지로부터 region을 추출해내는 과정이다.   
논문들에서는 selective search 방법이라는 비슷한 질감이나 색, 강도를 가진 인접한 픽셀들을 연결하여 바운 박스를 구성하는 방법을 사용한다.   
selective search으로 생성한 바운딩 박스의 output을 다시 CNN 레이어의 인푹으로 활용하게 되고, input으로 변환하는 과정에서 warp을 하게된다.   
   
#### (2) CNN : feature vector extract   
앞에서 생성한 region을 압축한 input으로 하여, pre-trained된 CNN 모듈에 통과시켜 새로운 feature를 생성한다.    
논문에서는 AlexNet, VggNet을 기반으로 튜닝 과정을 거친 CNN모델을 사용하였다.   
기존의 CNN Classifier에서 최종 Softmax 분류기 부분을 제외한 output을 결과물 feature로 사용한다.   
데이터 중에 정체를 알 수 없는 image feature들이 존재하는데 대부분은 이 feature를 사용한다.   
결과물 feature를 fixed-length feature vector라고 부른다.   
   
#### (3) Classify Algorithm   
fixed-length feature vector를 input으로 하는 분류기를 마지막에 만들어준다.  
결과물을 더욱 tight한 bounding box를 만들면 성능을 꽤 높은 수준으로 높여준다.   
하지만 필수적인 단계는 아니다.

#### (4) 요약   
1. selective search 방법으로 image로 부터 Object가 존재할 적절한 region을 추출하기 위해 bounding box 설정   
2. region을 AlexNet, VggNet의 기반의 CNN 모듈에 통과한다.   
3. classify regions을 통해 clasifier와 bounding box regressor로 처리   
   
#### (5) 단점   
모든 proposal에 대해 CNN을 거쳐야 하므로 연산량이 매우 많다.   
   
### 3. Fast R-CNN   
Fast R-CNN은 모든 proposal이 네트워크를 커쳐야 하는 R-CNN의 bottleneck 구조의 단점을 개선하고 제안된 방식이다.   
R-CNN과 가장 큰 차이점은, 각 proposal들이 CNN을 거치는 것이 아니라 전체 이미지에 대해 CNN을 한번 거친 후 출력된 feature map 단계에서 객체 탐지를 수행한다는 것이다.   
   
#### (1)


### 참고   
1. https://yamalab.tistory.com   
2. https://seongkyun.github.io/papers/2019/01/06/Object_detection/   
3. https://yamalab.tistory.com/113   

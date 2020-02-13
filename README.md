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
   
![fastRCNN](https://user-images.githubusercontent.com/59756209/74311472-ecc3e500-4db2-11ea-9ca8-1cfbcfeb1111.PNG)   
   
Fast R-CNN은 모든 proposal이 네트워크를 커쳐야 하는 R-CNN의 bottleneck 구조의 단점을 개선하고 제안된 방식이다.   
R-CNN과 가장 큰 차이점은, 각 proposal들이 CNN을 거치는 것이 아니라 전체 이미지에 대해 CNN을 한번 거친 후 출력된 feature map 단계에서 객체 탐지를 수행한다는 것이다.   
   
#### (1) Rol Pooling(region of interst pooling)   
모든 Rol Projection마다 convolution 연산을 하는 대신, 입력 이미지에 한번만 CNN을 적용하고 Rol Pooling으로 객체를 판별하기 위한 특징을 추출하는 것이다.   
Fast R-CNN은 입력 이미지에 대한 한번만 연산된 feature map에 Rol Pooling을 적용시키는 구조를 가지게 된다.   
R-CNN이 selective search * CNN 의 연산량이라면 Fast R-CNN은 (selecitve search + Rol Pooling) * 1의 연산량이 되는 것이다.   
이로 인해 연산량이 급격하게 줄어들었다.   
Rol Pooling은 어떤 output size가 와도 같은 사이즈로 통일시켜주는 역할을 수행한다.   
RP이후에는 검출된 object의 클래스를 분류하는 softmax 분류기와 bounding box를 추정하는 bbox regressor를 학습한다.   
   
#### (2) 요약   
R-CNN + Rol Pooling (연산속도가 업그레이드)   
   
#### (3) 단점   
하지만 여전히, Rol Projection을 생성하는 시간은 오래걸린다.   
selective seach를 수행하는 region proposal 부분이 외부에 존재하여 inference에서 bottleneck을 일으킨다.   
   
### 4. Faster R-CNN   
   
![fasterRCNN](https://user-images.githubusercontent.com/59756209/74398940-7bd90780-4e5c-11ea-9dc9-2ae6dc344249.PNG)   
   
selective search없이 region proposal network을 학습하는 구조로 개선시킨 모델이다.   
RPN(region proposal network)는 feature map을 input으로, RP(Rol Pooling)을 output으로 하는 네트워크으로써, selective search의 열할을 대체한다.   

#### (1) 1x1 convolution   
1x1 convolution은 pointwise convolution이라고도 한다.   
채널을 여러개가진 feature map에 대한 채널 축소이다.   
예로써, 32 x 32 x 512 feature map이 있다고 할때, 1x1 convolution을 거치게 되면 32 x 32 x 128이나 32 x 32 x 1024 와 같은 채널 부분의 parameter size를 변경하는 것이 가능하다.   
즉, 각 픽셀에 대해 input image parameter와 filter parameter 간의 fully conneted layer와 같은 형태가 된다.   
이를 통해 CNN layer에서 채널 단위의 축소 또는 확장 기능을 넣어주는 것이다.   
   
#### (2) RPN(region proposal network)   
RPN의 input은 feature map(shared network인 CNN의 output)이다.   
input feature map에 3 x 3 x 256-d filter로 한번 더 convolution을 해준다. 이 결과, feature map과 동일한 size의 feature map이 된다.   
예를 들면, 10 x 10 x 256-d 의 feature map이 input 이라면 3 x 3 x 256-d, 1 stride 1 padding filter를 적용하여 10 x 10 x 256-d의 output으로 도출하는 것이다.   
이 convolution 과정에서 각 지점(anchor)마다 RP를 구한다.   
output feature map을 1x1 convolution 을 수행하여 2개의 output을 도출한다.   
2개의 output은 각각 10 x 10 x (9 x 2) 와 10 x 10 x (9 x 4)의 사이즈이다.   
(9 x 2)는 지점(anchor) * class 이고, (9 x 4)는 지점(anchor) * 4개의 좌표를 의미한다.   
   
#### (3) anchor (지점)   
생성된 2개의 output은 각각 '물체인지 아닌지 판별' 과 'bb box를 예측'하는 용도로 사용된다.   
anchor는 미리 정의된 reference bounding box이다.   
다양한 크기와 비율로 n개의 anchor를 미리 정의하고 3 x 3 filter로 sliding window(convolution)를 할때, sliding마다 n개의 bounding box 후보를 생성하는 것이다.   

### 참고   
1. https://yamalab.tistory.com   
2. https://seongkyun.github.io/papers/2019/01/06/Object_detection/   
3. https://mylifemystudy.tistory.com/82   

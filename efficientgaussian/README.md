# Improve EAGLES
### Efficient Accelerated 3D Gaussians with Lightweight Encodings

EAGLES 프로젝트는 임베디드 환경에서의 3D 렌더링 효율성을 개선하기 위해 모델의 경량화와 최적화를 목표로 합니다. 본 프로젝트는 EAGLES 논문을 기반으로 진행되었으며, 추론 시간 단축 및 저장 메모리 감소를 중점으로 성능 개선을 시도하였습니다.

## 📌 프로젝트 개요

- **목표**: 제한된 임베디드 자원에서 실시간 3D 렌더링 가능성을 증대.
- **핵심 개선 사항**:
  - 기존 EAGLES 모델의 가지치기 방식을 동적으로 조정하여 초기 경량화와 후반 품질 재구성 강화.
  - 불필요하거나 중복된 Gaussian 포인트를 효과적으로 제거하여 효율성을 극대화.

## 🛠 주요 기능 및 성과

### 파일 크기 및 성능 비교

| Metric                | EAGLES      | Improve     | 개선 비율     |
|-----------------------|-------------|-------------|---------------|
| **Ply 파일 크기**      | 49.3 MB     | 21.5 MB     | **56% 감소**  |
| **PKL 파일 크기**      | 9.28 MB     | 4.08 MB     | **56% 감소**  |
| **SSIM (품질 지표)**   | 0.9250      | 0.9221      | -             |
| **PSNR (품질 지표)**   | 28.556      | 28.207      | -             |
| **LPIPS (오차 지표)**  | 0.2012      | 0.2102      | -             |
| **학습 시간**          | 334.99 sec  | 286.88 sec  | **14.36% 단축** |
| **렌더링 시간**        | 60.50 sec   | 48.35 sec   | **20.08% 단축** |

> Ply 및 PKL 파일 크기를 56% 감소시키고, 학습 및 렌더링 시간을 각각 14.36%와 20.08% 단축하는 데 성공했습니다.

## 🔍 개선 과정

### [다양한 접근과 분석으로 56%의 경량화 개선에 성공하다]

EAGLES 프로젝트는 3D 렌더링 관련 산학 협력 프로젝트로 시작되었습니다. 목표는 기존 3D Gaussian Splatting(3DGS) 모델의 최적화를 기반으로, 추가적인 성능 개선을 이루는 것이었습니다. ‘EAGLES’ 논문을 중심으로 최적화 방법을 탐구하며, 여러 검증을 통해 높은 수준의 최적화를 이룬 기술을 개선해야 하는 도전 과제에 직면했습니다.

1. **최적화 전략 탐구**  
   논문에서 사용된 Pruning(가지치기) 방법이 학습 후반에는 품질 유지 때문에 적용되지 않는다는 점과, 정적으로만 적용된다는 점을 개선 대상으로 설정했습니다.  
   - 초기 Pruning을 강화하여 경량화에 집중하고, 후반부에는 품질 재구성을 강화하는 방식으로 접근했습니다.
   - 초기 Pruning 비율을 높이고 적용 간격을 동적으로 조정하는 새로운 방식을 도입했습니다.

2. **문제 해결 및 시뮬레이션**  
   예상과 달리, 수정된 모델은 더 많은 메모리 저장 공간을 요구하는 문제가 발생했습니다.  
   - 학습 로그를 분석한 결과, 극초기 Pruning 적용 시 누적 Gaussian 포인트 부족으로 인해 의미 있는 경량화로 이어지지 않는 것을 확인했습니다.
   - 이를 해결하기 위해 Pruning 적용 시점을 학습 일정 단계 이후로 조정하고, 동적 간격 계산식을 테스트 케이스로 구현하여 최적의 결과를 도출했습니다.

3. **최종 결과**  
   다양한 접근과 시뮬레이션 끝에 기존 3DGS 대비 50% 압축된 EAGLES 모델에서 추가로 56%의 경량화와 20% 추론 시간 단축을 달성했습니다.  
   - PSNR, SSIM 등 주요 품질 지표를 유지하며 모델 성능을 더욱 향상했습니다.

## 💻 주요 코드

아래는 동적인 interval과 ratio 조정 코드입니다.

```python
# Dynamic pruning interval calculation
def get_dynamic_prune_interval(iteration):
    max_interval = 700  # 최대 간격
    min_interval = 100  # 최소 간격
    # 동적 Pruning 간격 계산
    return max(min_interval, min(max_interval, int(opt.infl_prune_interval + 
            (max_interval - opt.infl_prune_interval) * (iteration / opt.prune_until_iter))))

# Dynamic quantile threshold calculation
def get_dynamic_quantile_threshold(iteration):
    min_threshold = 0.01  # 최소 임계값
    # 동적 임계값 계산
    return max(min_threshold, opt.quantile_threshold - 
               (opt.quantile_threshold - min_threshold) * (iteration / opt.prune_until_iter))

# 변수 정의
prune_start_iter = 1500  # Pruning 시작 Iteration
next_prune_iteration = prune_start_iter  # 초기 Pruning 간격 설정

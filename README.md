# binary_Classification_using_pytorch_with_ray_tune
Training a Dataset on a Local Environment Using PyTorch and Ray Tune[Detail Code](https://github.com/qkrwoghd04/binary_Classification_using_pytorch_with_ray_tune/blob/main/ipynb/vit_classification_using_pytorch.ipynb)

---
### Installation(colab)
**PyTorch** : 강력한 GPU 가속 기능을 갖춘 텐서 계산을 지원 similar with numpy<br>
**Torchinfo** : 모델 레이어, 매개변수, 연결 관계등 모델 구조를 시각적으로 표현<br>
**Ray** : 분산 훈련을 통해 다양한 파라미터간에 가장 좋은 조합을 파악 가능<br>

```
!pip install torch
!pip install torchinfo
!pip install ray 
!pip install -U tensorboardx 
```
---
### Load dataset
```
file_path = r"/content/drive/MyDrive/image_dataset/processed/"
train_data_csv = "train_captions.csv"
test_data_csv = "test_captions.csv"
# transform 함수를 사전에 구현
transform = image_transform() 
test_transform = test_img_transform()

# CustomImageDataset() 함수를 사전에 구현
train_dataset = CustomImageDataset(csv_file=train_data_csv, img_dir=file_path, transform=transform,) 
# 트레이닝과 검증 데이터셋 크기 계산
total_train = len(train_dataset)
val_size = int(0.20 * total_train)
train_size = total_train - val_size

# 데이터셋 분할
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
test_dataset = CustomImageDataset(csv_file=test_data_csv, img_dir=file_path, transform=test_transform)

train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=1)
val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=1)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
```
---
### Load pretrained model
```
model = timm.create_model('vit_base_patch16_224', pretrained=True)
config = resolve_data_config({}, model=model)
transform = create_transform(**config)
summary(model, input_size=(batch_size, IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT))
```
---
### Ray tune
Ray tune 함수에 상단 부분은 model 과 loss 등등 변수들을 정의하고, 중간 부분에서 train_dataloader 와 val_dataloader를 통해서 학습을 진행합니다. 그리고 마지막으로 checkpoint를 저장하는 것으로 끝이납니다.
> **Tuner를 정의할때 사용할 함수 정의**
```
CHECKPOINT_FREQ = 3
def train_func(config):
  start = 1
  device = "cuda" if torch.cuda.is_available() else "cpu"
  # model = vit_model().to(device)
  model = timm.create_model('vit_base_patch16_224', pretrained=True)
  model.to("cuda")
  model_config = resolve_data_config({}, model=model)
  transform = create_transform(**model_config)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

  # #Define checkpoint
  checkpoint = train.get_checkpoint()
  if checkpoint:
      with checkpoint.as_directory() as checkpoint_dir:
          checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
          print(checkpoint_dict)
          start = checkpoint_dict["epoch"] + 1
          model.load_state_dict(checkpoint_dict["model_state"])

  train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=1)
  val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=1)

  #model training
  for epoch in range(start, config["epochs"] + 1):  # loop over the dataset multiple times
      running_loss = 0.0
      epoch_steps = 0
      for i, data in enumerate(train_dataloader, 0):
          # get the inputs; data is a list of [inputs, labels]
          inputs, labels = data
          inputs, labels = inputs.to(device), labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward + backward + optimize
          outputs = model(inputs)
          loss = loss_fn(outputs, labels)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          epoch_steps += 1
          if i % 2000 == 1999:  # print every 2000 mini-batches
              print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
              running_loss = 0.0

      # Validation loss
      val_loss = 0.0
      val_steps = 0
      total = 0
      correct = 0
      for i, data in enumerate(val_dataloader, 0):
          with torch.no_grad():
              inputs, labels = data
              inputs, labels = inputs.to(device), labels.to(device)

              outputs = model(inputs)
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()

              loss = loss_fn(outputs, labels)
              val_loss += loss.cpu().numpy()
              val_steps += 1
      metrics = {
            "loss": running_loss / len(train_dataloader),
            "val_loss": val_loss / len(val_dataloader),
            "mean_accuracy": correct / total
      }
      with tempfile.TemporaryDirectory() as tempdir:
          print(os.path.join(tempdir, "checkpoint.pt"))
          torch.save(
              {"epoch": epoch, "model_state": model.state_dict()},
              os.path.join(tempdir, "checkpoint.pt"),
          )
          train.report(metrics=metrics, checkpoint=Checkpoint.from_directory(tempdir))
```

> **Ray tune으로 학습시키기**
```
storage_path = "/content/drive/MyDrive/ray_results" # checkpoint 파일들이 저장될 위치
exp_name = "tune_analyzing_results" # 폴더 이름
trainable_with_resources = tune.with_resources(train_func, {"cpu":4 , "gpu":1, "accelerator_type:T4":1}) # 사용할 train 함수와 cpu 및 gru 정의
# tuner를 관리하는 scheduler
scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)
# tuner 정의
tuner = tune.Tuner(
    trainable_with_resources,
    param_space={
        "lr": tune.loguniform(1e-4 , 4e-4, 1e-5),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": 10,
        "should_checkpoint":True,
    },
    run_config=train.RunConfig(
        name=exp_name,
        # stop={"training_iteration": 3},
        checkpoint_config=train.CheckpointConfig(
            checkpoint_score_attribute="mean_accuracy",
            num_to_keep=3
        ),
        storage_path=storage_path,
    ),
    tune_config=tune.TuneConfig(mode="min", metric="val_loss", num_samples=8, max_concurrent_trials=2, scheduler=scheduler),
)
result_grid: ResultGrid = tuner.fit() # 학습시작

```

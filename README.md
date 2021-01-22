# PyTorch Android Example

This application runs TorchScript serialized modes on static image and random tensors needed for benchmarking.

#### 1. Model Preparation

Follow src/mobile/Mobile.ipynb for mobile optimizing a script model

#### 2. Cloning from github
```
git clone https://github.com/pytorch/android-demo-app.git
cd HelloWorldApp
```
If [Android SDK](https://developer.android.com/studio/index.html#command-tools) and [Android NDK](https://developer.android.com/ndk/downloads) are already installed you can install this application to the connected android device or emulator with:
```
./gradlew installDebug
```

We recommend you to open this project in [Android Studio 3.5.1+](https://developer.android.com/studio) (At the moment PyTorch Android and demo applications use [android gradle plugin of version 3.5.0](https://developer.android.com/studio/releases/gradle-plugin#3-5-0), which is supported only by Android Studio version 3.5.1 and higher),
in that case you will be able to install Android NDK and Android SDK using Android Studio UI.

## Update this application to support FLM usecase

#### 3. Gradle dependencies

Pytorch android is added to the HelloWorld as [gradle dependencies](https://github.com/pytorch/android-demo-app/blob/master/HelloWorldApp/app/build.gradle#L28-L29) in build.gradle:

```
repositories {
    jcenter()
}

dependencies {
    implementation 'org.pytorch:pytorch_android:1.7.1'
    implementation 'org.pytorch:pytorch_android_torchvision:1.7.1'
}
```
Where `org.pytorch:pytorch_android` is the main dependency with PyTorch Android API, including libtorch native library for all 4 android abis (armeabi-v7a, arm64-v8a, x86, x86_64).
Further in this doc you can find how to rebuild it only for specific list of android abis. 

`org.pytorch:pytorch_android_torchvision` - additional library with utility functions for converting `android.media.Image` and `android.graphics.Bitmap` to tensors.

#### 4. Find the images and models in assets

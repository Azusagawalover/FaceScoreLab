plugins { alias(libs.plugins.android.application) }
// 独立测试包使用不同的包名和数据目录。
val verificationIdSuffix: String? = (findProperty("facescorelabApplicationIdSuffix") as String?)
android {
    namespace = "com.example.facescorelab"
    compileSdk = 34
    defaultConfig {
        applicationId = "com.example.facescorelab"
        minSdk = 24
        targetSdk = 34
        versionCode = 16
        versionName = "3.4.1"
        if (verificationIdSuffix != null) applicationIdSuffix = verificationIdSuffix
        manifestPlaceholders["facescorelabApplicationLabel"] = if (verificationIdSuffix != null) "FaceScore 验证" else "FaceScore Lab"
        // MediaPipe 0.10.14 不提供 x86_64 JNI，限制为完整可用的 ABI。
        ndk { abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86") }
        testInstrumentationRunner = "com.example.facescorelab.ContractInstrumentation"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    androidResources { noCompress += listOf("task", "tflite", "onnx") }
}
dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.activity)
    implementation(libs.constraintlayout)
    implementation("com.google.mediapipe:tasks-vision:0.10.14")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")
}

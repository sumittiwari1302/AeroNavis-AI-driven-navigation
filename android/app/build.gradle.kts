plugins {
    alias(libs.plugins.androidApplication) apply true
    alias(libs.plugins.kotlinAndroid) apply true
    alias(libs.plugins.kotlinSerialization) apply true
    alias(libs.plugins.kotlinParceable) apply true
}

android {
    namespace = "com.navx"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.navx"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        
        // No network access at runtime
        setProperty("android.net.INTERNET_PERMISSION", "false")
    }

    buildFeatures {
        viewBinding = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf("-Xopt-in=kotlin.RequiresOptIn")
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
        debug {
            isDebuggable = true
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packagingOptions {
        resources {
            excludes += "/META-INF/*"
        }
        jniLibs {
            pickFirsts += listOf(
                "libc++_shared.so",
                "libtensorflowlite_jni.so",
                "libtensorflowlite_gpu_jni.so"
            )
        }
    }

    packaging {
        resources {
            excludes += "/META-INF/*"
        }
    }
}

dependencies {
    implementation(libs.coreKtx)
    implementation(libs.lifecycleRuntime)
    implementation(libs.lifecycleViewModel)
    implementation(libs.material3)
    implementation(libs.maplibre)
    
    // TensorFlow Lite
    implementation(libs.tflite)
    implementation(libs.tfliteGpu)
    
    // Math library
    implementation(libs.multik)
    
    // Coroutines
    implementation(libs.kotlinxCoroutinesCore)
    implementation(libs.kotlinxCoroutinesAndroid)
    implementation(libs.kotlinxSerialization)
    
    // CameraX
    implementation(libs.cameraXCore)
    implementation(libs.cameraXCamera2)
    implementation(libs.cameraXView)
    implementation(libs.cameraXLifecycle)
    
    // TensorFlow Lite
    implementation(libs.tflite)
    implementation(libs.tfliteGpu)
    
    // Room
    implementation(libs.roomRuntime)
    kapt(libs.roomCompiler)
    
    // OKHttp for any needed networking (though offline mode enforced)
    implementation(libs.okhttp)
    
    // Testing
    testImplementation(libs.junit)
    testImplementation(libs.junitJupiter)
    testImplementation(libs.truth)
    testImplementation(libs.robolectric)
    
    androidTestImplementation(libs.junit)
    androidTestImplementation(libs.espressoCore)
    androidTestImplementation(libs.junitRunner)
    androidTestImplementation(libs.junitRules)
}

kapt {
    correctErrorTypes = true
}

tasks.withType(org.jetbrains.kotlin.gradle.tasks.KotlinCompile).configureEach {
    kotlinOptions {
        freeCompilerArgs = listOf("-Xopt-in=kotlin.RequiresOptIn")
    }
}

tasks.named("check") {
    dependsOn("testDebugUnitTest")
}
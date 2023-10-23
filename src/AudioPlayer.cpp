#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <portaudio.h>

#include "AudioPlayer.hpp"

AudioPlayer::AudioPlayer()
{
}

AudioPlayer::~AudioPlayer()
{
    // Terminate PortAudio
    err_ = Pa_Terminate();

    // Close audio file
    sf_close(file_);
}

void AudioPlayer::init(const char *filename)
{
    // Open audio file
    file_ = sf_open(filename, SFM_READ, &sfinfo_);
    if (!file_)
    {
        std::cerr << "Could not open file " << filename << std::endl;
        return;
    }

    // Initialize PortAudio
    err_ = Pa_Initialize();
    if (err_ != paNoError)
    {
        std::cerr << "PortAudio initialization error: " << Pa_GetErrorText(err_) << std::endl;
        sf_close(file_);
        return;
    }
}

void AudioPlayer::play()
{
    std::lock_guard<std::mutex> lock(audioMutex_);
    // Open the default audio stream
    err_ = Pa_OpenDefaultStream(&stream_, 0, sfinfo_.channels, paFloat32, sfinfo_.samplerate, framesPerBuffer_, NULL, NULL);
    if (err_ != paNoError)
    {
        std::cerr << "PortAudio stream opening error: " << Pa_GetErrorText(err_) << std::endl;
        Pa_Terminate();
        sf_close(file_);
        return;
    }

    // Start the audio stream
    err_ = Pa_StartStream(stream_);
    if (err_ != paNoError)
    {
        std::cerr << "PortAudio stream starting error: " << Pa_GetErrorText(err_) << std::endl;
        Pa_CloseStream(stream_);
        Pa_Terminate();
        sf_close(file_);
        return;
    }

    // Allocate buffers
    buffer_ = new float[framesPerBuffer_ * sfinfo_.channels];
    readBuffer_ = new float[framesPerBuffer_ * sfinfo_.channels];
    writeBuffer_ = new float[framesPerBuffer_ * sfinfo_.channels];

    // Audio loop
    while (!stopPlaying_ && (framesRead_ = sf_readf_float(file_, buffer_, framesPerBuffer_)) > 0)
    {
        // Deep copy audio buffer to double buffers for external access
        {
            std::lock_guard<std::mutex> lock(bufferMutex_);
            std::copy(buffer_, buffer_ + framesPerBuffer_ * sfinfo_.channels, writeBuffer_);

            // Swap buffers
            float *temp = readBuffer_;
            readBuffer_ = writeBuffer_;
            writeBuffer_ = temp;
        }

        // Write audio buffer to the audio stream
        err_ = Pa_WriteStream(stream_, buffer_, framesRead_);
        if (err_ != paNoError)
        {
            std::cerr << "PortAudio stream writing error: " << Pa_GetErrorText(err_) << std::endl;
            Pa_CloseStream(stream_);
            Pa_Terminate();
            sf_close(file_);
            return;
        }
    }

    // Clean up
    delete[] buffer_;
    delete[] readBuffer_;
    delete[] writeBuffer_;
    buffer_ = nullptr;
    readBuffer_ = nullptr;
    writeBuffer_ = nullptr;
}

void AudioPlayer::stop()
{
    // Set stopPlaying_ to true to escape the audio loop in AudioPlayer::play, which unlocks audioMutex_
    stopPlaying_ = true;
    std::lock_guard<std::mutex> lock(audioMutex_);

    // Stop the audio stream
    err_ = Pa_StopStream(stream_);
    if (err_ != paNoError)
    {
        std::cerr << "PortAudio stream stopping error: " << Pa_GetErrorText(err_) << std::endl;
    }

    // Close the audio stream
    err_ = Pa_CloseStream(stream_);
    if (err_ != paNoError)
    {
        std::cerr << "PortAudio stream closing error: " << Pa_GetErrorText(err_) << std::endl;
        Pa_Terminate();
        sf_close(file_);
        return;
    }
}

bool AudioPlayer::isPlaying()
{
    return !stopPlaying_;
}

int AudioPlayer::getAudioBufferSize()
{
    return framesPerBuffer_ * sfinfo_.channels;
}

void AudioPlayer::copyAudioBufferData(float *dest, int size)
{
    // If audio is not playing, return zeros
    if (stopPlaying_)
        std::fill(dest, dest + size, 0.0f);

    // Copy audio buffer to dest
    std::lock_guard<std::mutex> lock(bufferMutex_);
    std::copy(readBuffer_, readBuffer_ + size, dest);
}

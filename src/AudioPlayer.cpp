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

    // Play the audio file
    float buffer[framesPerBuffer_ * sfinfo_.channels];
    while (!stopPlaying_ && (framesRead_ = sf_readf_float(file_, buffer, framesPerBuffer_)) > 0)
    {
        err_ = Pa_WriteStream(stream_, buffer, framesRead_);
        if (err_ != paNoError)
        {
            std::cerr << "PortAudio stream writing error: " << Pa_GetErrorText(err_) << std::endl;
            Pa_CloseStream(stream_);
            Pa_Terminate();
            sf_close(file_);
            return;
        }
    }
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
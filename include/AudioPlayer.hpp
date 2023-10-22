#ifndef AUDIOPLAYER_HPP
#define AUDIOPLAYER_HPP

#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>

#include <sndfile.h>
#include <portaudio.h>

class AudioPlayer
{
public:
    const int framesPerBuffer_ = 512;

    const char *filename_;
    SNDFILE *file_;
    SF_INFO sfinfo_;
    PaError err_;
    PaStream *stream_;
    sf_count_t framesRead_;

    // audioMutex_ ensured the AudioPlayer::play and AudioPlayer::stop are thread-safe
    std::mutex audioMutex_;
    // Setting stopPlaying_to true will escape the audio loop in AudioPlayer::play
    std::atomic<bool> stopPlaying_ = false;

    AudioPlayer();
    ~AudioPlayer();

    void init(const char *filename);
    void play();
    void stop();
};

#endif // AUDIOPLAYER_HPP

// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonSTT

namespace Season.STT;

/// <summary>
/// Loaded Whisper model sessions that can run repeated transcription without
/// reloading the ONNX files from disk.
/// </summary>
public sealed class WhisperContext : IDisposable
{
    readonly string modelDirectory;
    readonly Whisper.Config config;
    readonly InferenceSession encoderSession;
    readonly InferenceSession decoderSession;

    WhisperContext(string modelDirectory, Whisper.Config config, InferenceSession encoderSession, InferenceSession decoderSession)
    {
        this.modelDirectory = modelDirectory;
        this.config = config;
        this.encoderSession = encoderSession;
        this.decoderSession = decoderSession;
    }

    /// <summary>
    /// Loads the Whisper config and creates the encoder/decoder sessions.
    /// </summary>
    /// <param name="jsonPath">Path to `season-whisper.json`.</param>
    /// <param name="encoderPath">Path to the encoder ONNX model.</param>
    /// <param name="encoderDataPath">Path to the encoder external data; must sit next to <paramref name="encoderPath"/>.</param>
    /// <param name="decoderPath">Path to the decoder ONNX model.</param>
    /// <param name="provider">ONNX Runtime execution provider, for example <c>cpu</c>.</param>
    public static WhisperContext Load(string jsonPath, string encoderPath, string encoderDataPath, string decoderPath, string provider = "cpu")
    {
        if (string.IsNullOrWhiteSpace(jsonPath))
            throw new ArgumentException("Whisper config path is required.", nameof(jsonPath));
        if (string.IsNullOrWhiteSpace(encoderPath))
            throw new ArgumentException("Whisper encoder path is required.", nameof(encoderPath));
        if (string.IsNullOrWhiteSpace(decoderPath))
            throw new ArgumentException("Whisper decoder path is required.", nameof(decoderPath));

        // ONNX Runtime resolves external data relative to the model file, so the
        // encoder weights must sit next to the encoder graph.
        if (!string.IsNullOrWhiteSpace(encoderDataPath))
        {
            var encoderDir = Path.GetDirectoryName(Path.GetFullPath(encoderPath));
            var dataDir = Path.GetDirectoryName(Path.GetFullPath(encoderDataPath));

            if (!string.Equals(encoderDir, dataDir, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"Encoder external data '{encoderDataPath}' must sit in the same directory as '{encoderPath}'.");
            }
        }

        var config = Whisper.LoadConfig(jsonPath);

        config.Provider = string.IsNullOrWhiteSpace(provider) ? "cpu" : provider;

        var encoderSession = Whisper.CreateSession(encoderPath, config.Provider);

        try
        {
            var decoderSession = Whisper.CreateSession(decoderPath, config.Provider);

            return new WhisperContext(Path.GetDirectoryName(Path.GetFullPath(encoderPath)), config, encoderSession, decoderSession);
        }
        catch
        {
            encoderSession.Dispose();

            throw;
        }
    }

    /// <summary>
    /// Runs speech-to-text inference with the already loaded sessions.
    /// </summary>
    /// <param name="voice">WAV PCM16 audio bytes.</param>
    /// <param name="defaultLanguage">Default language code, for example <c>en</c>.</param>
    /// <returns>The decoded transcription text.</returns>
    public string Detect(byte[] voice, string defaultLanguage)
    {
        config.DefaultLanguage = string.IsNullOrWhiteSpace(defaultLanguage) ? "" : defaultLanguage;

        var audio = Whisper.ResampleToTargetRate(
            Whisper.ReadWavPcm16(voice, out int sourceSampleRate),
            sourceSampleRate,
            config.SampleRate,
            config.MaxSamples);

        var inputFeatures = Whisper.CreateInputFeatures(modelDirectory, audio, config);

        using var encoderResults = encoderSession.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(config.EncoderInput, inputFeatures)
        });
        var encoderHiddenStates = Whisper.CopyFloatTensor(encoderResults, config.EncoderOutput);

        string? language = string.IsNullOrWhiteSpace(config.Language)
            ? (string.IsNullOrWhiteSpace(config.DefaultLanguage) ? null : config.DefaultLanguage)
            : config.Language;

        var tokenIds = new List<long> { config.Sot };
        if (!string.IsNullOrWhiteSpace(language) &&
            config.Languages.TryGetValue(language, out long languageToken))
        {
            tokenIds.Add(languageToken);
        }

        tokenIds.Add(config.Transcribe);
        tokenIds.Add(config.NoTime);
        int promptTokenCount = tokenIds.Count;

        for (int i = 0; i < config.MaxTokens; i++)
        {
            var decoderInputIds = new DenseTensor<long>(tokenIds.ToArray(), new[] { 1, tokenIds.Count });
            using var decoderResults = decoderSession.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(config.DecoderInput, decoderInputIds),
                NamedOnnxValue.CreateFromTensor(config.DecoderEncoderInput, encoderHiddenStates)
            });

            int nextToken = Whisper.ArgMaxLastLogit(Whisper.GetTensor(decoderResults, config.DecoderOutput).AsTensor<float>());
            if (nextToken == config.Eot)
                break;

            tokenIds.Add(nextToken);
        }

        return Whisper.DecodeTokens(tokenIds.Skip(promptTokenCount), config).Trim();
    }

    public void Dispose()
    {
        decoderSession.Dispose();
        encoderSession.Dispose();
    }
}

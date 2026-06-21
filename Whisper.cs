// Copyright (c) SeasonEngine and contributors.
// Licensed under the MIT License.
// https://github.com/SeasonRealms/SeasonSTT

namespace SeasonSTT;

/// <summary>
/// Provides a simple Whisper ONNX inference entry point for WAV PCM16 audio.
/// </summary>
public static class Whisper
{
    /// <summary>
    /// Runs speech-to-text inference with a Whisper ONNX model bundle.
    /// </summary>
    /// <param name="model">Directory that contains the Whisper model files and `season-whisper.json`.</param>
    /// <param name="voice">WAV PCM16 audio bytes.</param>
    /// <param name="defaultLanguage">Default language code, for example <c>en</c>.</param>
    /// <returns>The decoded transcription text.</returns>
    public static string Detect(string model, byte[] voice, string defaultLanguage)
    {
        var config = LoadConfig($"{model}/season-whisper.json");

        config.Provider = "cpu";

        config.DefaultLanguage = defaultLanguage;

        var audio = ResampleToTargetRate(
            ReadWavPcm16(voice, out int sourceSampleRate),
            sourceSampleRate,
            config.SampleRate,
            config.MaxSamples);

        using var encoderSession = CreateSession(model, config.EncoderModel, config.Provider);
        using var decoderSession = CreateSession(model, config.DecoderModel, config.Provider);
        var inputFeatures = CreateInputFeatures(model, audio, config);

        using var encoderResults = encoderSession.Run(new[]
        {
            NamedOnnxValue.CreateFromTensor(config.EncoderInput, inputFeatures)
        });
        var encoderHiddenStates = CopyFloatTensor(encoderResults, config.EncoderOutput);

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

            int nextToken = ArgMaxLastLogit(GetTensor(decoderResults, config.DecoderOutput).AsTensor<float>());
            if (nextToken == config.Eot)
                break;

            tokenIds.Add(nextToken);
        }

        return DecodeTokens(tokenIds.Skip(promptTokenCount), config).Trim();
    }

    static Config LoadConfig(string path)
    {
        var bytes = File.ReadAllBytes(path); // DeviceServices.Core.LoadFile(path);

        return JsonSerializer.Deserialize<Config>(
            bytes,
            new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidDataException($"Failed to read Whisper config: {path}");
    }

    static DenseTensor<float> CreateInputFeatures(string modelDirectory, float[] audio, Config config)
    {
        if (!string.IsNullOrWhiteSpace(config.PreprocessModel))  //DeviceServices.Core.LoadFileExists($"{modelDirectory}/{config.PreprocessModel}"))
        {
            using var preprocessSession = CreateSession(modelDirectory, config.PreprocessModel, config.Provider);
            var waveform = new DenseTensor<float>(audio, new[] { 1, audio.Length });
            using var preprocessResults = preprocessSession.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(config.PreprocessInput, waveform)
            });
            return CopyFloatTensor(preprocessResults, config.PreprocessOutput);
        }

        return ComputeLogMelSpectrogram(audio, config);
    }

    static InferenceSession CreateSession(string modelDirectory, string fileName, string? provider)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        foreach (var methodName in GetProviderMethodNames(provider))
        {
            var method = typeof(SessionOptions).GetMethod(methodName, BindingFlags.Instance | BindingFlags.Public);
            if (method == null)
                continue;

            try
            {
                var args = method.GetParameters()
                    .Select(CreateDefaultValue)
                    .ToArray();
                method.Invoke(options, args);
                break;
            }
            catch
            {
                // Continue trying the next provider when the API is unavailable or native dependencies are missing.
            }
        }

        return new InferenceSession($"{modelDirectory}/{fileName}", options);

        //using var stream = DeviceServices.Core.LoadFile($"{modelDirectory}/{fileName}");
        //return new InferenceSession(stream.ReadAllBytes(), options);
    }

    static object? CreateDefaultValue(ParameterInfo parameter)
    {
        if (parameter.HasDefaultValue)
            return parameter.DefaultValue;

        return parameter.ParameterType.IsValueType
            ? Activator.CreateInstance(parameter.ParameterType)
            : null;
    }

    static string[] GetProviderMethodNames(string? provider)
    {
        return (provider ?? "auto").ToLowerInvariant() switch
        {
            "cpu" => Array.Empty<string>(),
            "dml" => new[] { "AppendExecutionProvider_DML" },
            "cuda" => new[] { "AppendExecutionProvider_CUDA" },
            "coreml" => new[] { "AppendExecutionProvider_CoreML" },
            "nnapi" => new[] { "AppendExecutionProvider_Nnapi", "AppendExecutionProvider_NNAPI" },
            _ when OperatingSystem.IsWindows() => new[] { "AppendExecutionProvider_DML", "AppendExecutionProvider_CUDA" },
            _ when OperatingSystem.IsAndroid() => new[] { "AppendExecutionProvider_Nnapi", "AppendExecutionProvider_NNAPI" },
            _ when OperatingSystem.IsIOS() || OperatingSystem.IsMacOS() => new[] { "AppendExecutionProvider_CoreML" },
            _ => new[] { "AppendExecutionProvider_CUDA" }
        };
    }

    static DenseTensor<float> CopyFloatTensor(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        string outputName)
    {
        var tensor = GetTensor(results, outputName).AsTensor<float>();
        return new DenseTensor<float>(tensor.ToArray(), tensor.Dimensions.ToArray());
    }

    static DenseTensor<float> ComputeLogMelSpectrogram(float[] audio, Config config)
    {
        int chunkSamples = config.MaxSamples;
        int nFft = config.NFft;
        int hopLength = config.HopLength;
        int nMels = config.NMels;
        int fftBins = nFft / 2 + 1;
        int frameCount = config.MaxFrames > 0 ? config.MaxFrames : chunkSamples / hopLength;

        if (config.MelFilters.Count != nMels * fftBins)
            throw new InvalidDataException($"Mel filter count mismatch. Expected {nMels * fftBins}, actual {config.MelFilters.Count}.");

        var padded = new float[chunkSamples + nFft];
        Array.Copy(audio, 0, padded, nFft / 2, Math.Min(audio.Length, chunkSamples));

        var window = BuildHannWindow(nFft);
        var powerSpectrum = new float[fftBins];
        var melSpectrum = new float[nMels * frameCount];

        for (int frame = 0; frame < frameCount; frame++)
        {
            int start = frame * hopLength;
            ComputePowerSpectrum(padded, start, window, nFft, powerSpectrum);

            for (int mel = 0; mel < nMels; mel++)
            {
                float sum = 0f;
                int melOffset = mel * fftBins;
                for (int bin = 0; bin < fftBins; bin++)
                    sum += config.MelFilters[melOffset + bin] * powerSpectrum[bin];

                melSpectrum[mel * frameCount + frame] = MathF.Log10(MathF.Max(sum, 1e-10f));
            }
        }

        float maxLog = melSpectrum.Length == 0 ? 0f : melSpectrum.Max();
        float minLog = maxLog - 8f;
        for (int i = 0; i < melSpectrum.Length; i++)
        {
            float clamped = MathF.Max(melSpectrum[i], minLog);
            melSpectrum[i] = (clamped + 4f) / 4f;
        }

        return new DenseTensor<float>(melSpectrum, new[] { 1, nMels, frameCount });
    }

    static float[] BuildHannWindow(int size)
    {
        var window = new float[size];
        if (size <= 1)
        {
            if (size == 1)
                window[0] = 1f;
            return window;
        }

        for (int i = 0; i < size; i++)
            window[i] = 0.5f - 0.5f * MathF.Cos((2f * MathF.PI * i) / (size - 1));
        return window;
    }

    static void ComputePowerSpectrum(float[] audio, int start, float[] window, int nFft, float[] output)
    {
        Array.Clear(output, 0, output.Length);
        int bins = output.Length;
        float scale = 2f * MathF.PI / nFft;

        for (int k = 0; k < bins; k++)
        {
            float real = 0f;
            float imag = 0f;
            for (int n = 0; n < nFft; n++)
            {
                float sample = audio[start + n] * window[n];
                float angle = scale * k * n;
                real += sample * MathF.Cos(angle);
                imag -= sample * MathF.Sin(angle);
            }
            output[k] = real * real + imag * imag;
        }
    }

    static DisposableNamedOnnxValue GetTensor(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results,
        string outputName)
    {
        return string.IsNullOrWhiteSpace(outputName)
            ? results.First()
            : results.First(result => result.Name == outputName);
    }

    static float[] ReadWavPcm16(byte[] bytes, out int sampleRate)
    {
        using var stream = new MemoryStream(bytes);
        using var reader = new BinaryReader(stream);

        if (new string(reader.ReadChars(4)) != "RIFF")
            throw new InvalidDataException("Whisper currently supports WAV/PCM16 input only.");

        reader.ReadInt32();
        if (new string(reader.ReadChars(4)) != "WAVE")
            throw new InvalidDataException("Whisper currently supports WAV/PCM16 input only.");

        sampleRate = 16000;
        short channels = 1;
        byte[] pcmBytes = Array.Empty<byte>();

        while (stream.Position + 8 <= stream.Length)
        {
            string chunkId = new string(reader.ReadChars(4));
            int chunkSize = reader.ReadInt32();
            long nextChunkPosition = stream.Position + chunkSize;

            if (chunkId == "fmt ")
            {
                short format = reader.ReadInt16();
                if (format != 1)
                    throw new InvalidDataException("Whisper currently supports PCM16 WAV only.");

                channels = reader.ReadInt16();
                sampleRate = reader.ReadInt32();
                stream.Position = nextChunkPosition;
            }
            else if (chunkId == "data")
            {
                pcmBytes = reader.ReadBytes(chunkSize);
            }
            else
            {
                stream.Position = nextChunkPosition;
            }

            if ((chunkSize & 1) != 0 && stream.Position < stream.Length)
                stream.Position++;
        }

        if (pcmBytes.Length == 0)
            throw new InvalidDataException("The WAV file does not contain a data chunk.");

        if (channels <= 0)
            throw new InvalidDataException("Invalid WAV channel count.");

        int frameCount = pcmBytes.Length / (channels * 2);
        var samples = new float[frameCount];

        for (int i = 0; i < frameCount; i++)
        {
            int mixed = 0;
            for (int channel = 0; channel < channels; channel++)
            {
                int offset = (i * channels + channel) * 2;
                mixed += BitConverter.ToInt16(pcmBytes, offset);
            }

            samples[i] = mixed / (32768f * channels);
        }

        return samples;
    }

    static float[] ResampleToTargetRate(float[] source, int sourceRate, int targetRate, int maxSamples)
    {
        if (sourceRate <= 0 || targetRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sourceRate), "Sample rates must be greater than 0.");

        if (sourceRate == targetRate)
        {
            if (source.Length >= maxSamples)
                return source.Take(maxSamples).ToArray();

            var padded = new float[maxSamples];
            Array.Copy(source, padded, source.Length);
            return padded;
        }

        int outputLength = Math.Min(maxSamples, (int)MathF.Round(source.Length * (float)targetRate / sourceRate));
        var output = new float[maxSamples];

        for (int i = 0; i < outputLength; i++)
        {
            float position = i * (sourceRate / (float)targetRate);
            int left = Math.Min((int)position, source.Length - 1);
            int right = Math.Min(left + 1, source.Length - 1);
            float weight = position - left;
            output[i] = source[left] + (source[right] - source[left]) * weight;
        }

        return output;
    }

    static int ArgMaxLastLogit(Tensor<float> logits)
    {
        var values = logits.ToArray();
        int vocabSize = logits.Dimensions[^1];
        int offset = values.Length - vocabSize;
        int bestIndex = 0;
        float bestValue = float.MinValue;

        for (int i = 0; i < vocabSize; i++)
        {
            float value = values[offset + i];
            if (value > bestValue)
            {
                bestValue = value;
                bestIndex = i;
            }
        }

        return bestIndex;
    }

    static string DecodeTokens(IEnumerable<long> tokenIds, Config config)
    {
        using var stream = new MemoryStream();

        foreach (long tokenId in tokenIds)
        {
            if (tokenId < 0 || tokenId >= config.TokenBytesBase64.Count)
                continue;

            if (tokenId >= config.TimeBegin)
                continue;

            string? tokenBase64 = config.TokenBytesBase64[(int)tokenId];
            if (string.IsNullOrEmpty(tokenBase64))
                continue;

            stream.Write(Convert.FromBase64String(tokenBase64));
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    sealed class Config
    {
        public string PreprocessModel { get; set; } = "preprocess_model.onnx";
        public string EncoderModel { get; set; } = "encoder_model.onnx";
        public string DecoderModel { get; set; } = "decoder_model.onnx";
        public string PreprocessInput { get; set; } = "waveform";
        public string PreprocessOutput { get; set; } = "input_features";
        public string EncoderInput { get; set; } = "input_features";
        public string EncoderOutput { get; set; } = "last_hidden_state";
        public string DecoderInput { get; set; } = "input_ids";
        public string DecoderEncoderInput { get; set; } = "encoder_hidden_states";
        public string DecoderOutput { get; set; } = "logits";
        public string? Provider { get; set; }
        public string DefaultLanguage { get; set; } = "";
        public string? Language { get; set; }
        public int SampleRate { get; set; } = 16000;
        public int MaxSamples { get; set; } = 480000;
        public int MaxFrames { get; set; } = 3000;
        public int NFft { get; set; } = 400;
        public int HopLength { get; set; } = 160;
        public int NMels { get; set; } = 80;
        public int MaxTokens { get; set; } = 128;
        public int Sot { get; set; }
        public int SotLang { get; set; }
        public int Transcribe { get; set; }
        public int NoTime { get; set; }
        public int Eot { get; set; }
        public int TimeBegin { get; set; }
        public Dictionary<string, long> Languages { get; set; } = new();
        public List<float> MelFilters { get; set; } = new();
        public List<string?> TokenBytesBase64 { get; set; } = new();
    }
}

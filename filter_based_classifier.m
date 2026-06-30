function filter_based_classifier()
    %% === STEP 0: Folder Paths and Setup ===
    base_path = pwd;
    train_amb = fullfile(base_path, 'train', 'ambulance');
    train_fire = fullfile(base_path, 'train', 'firetruck');
    test_amb = fullfile(base_path, 'test', 'ambulance');
    test_fire = fullfile(base_path, 'test', 'firetruck');
    fs = 44100; % Sample rate

    %% === STEP 1: Spectrum Analysis ===
    fprintf('Analyzing frequency spectrum...\n');
    avg_spec_amb = average_spectrum(train_amb, fs);
    avg_spec_fire = average_spectrum(train_fire, fs);

    % Plot both spectra
    freqs = linspace(0, fs/2, length(avg_spec_amb));
    figure;
    plot(freqs, avg_spec_amb, 'b', 'LineWidth', 1.5); hold on;
    plot(freqs, avg_spec_fire, 'r', 'LineWidth', 1.5);
    title('Average Frequency Spectrum');
    xlabel('Frequency (Hz)'); ylabel('Magnitude');
    legend('Ambulance', 'Firetruck'); grid on;

    %% === STEP 2: Filter Design ===
    fprintf('Designing filters...\n');
    filterA = designfilt('bandpassiir', 'FilterOrder', 8, ...
        'HalfPowerFrequency1', 500, 'HalfPowerFrequency2', 1000, ...
        'SampleRate', fs);
    filterB = designfilt('bandpassiir', 'FilterOrder', 8, ...
        'HalfPowerFrequency1', 1300, 'HalfPowerFrequency2', 2000, ...
        'SampleRate', fs);
    filterC = designfilt('bandpassiir', 'FilterOrder', 8, ...
        'HalfPowerFrequency1', 2000, 'HalfPowerFrequency2', 4000, ...
        'SampleRate', fs);

    %% === STEP 3: Compute Training Features ===
    amb_train_feats = compute_multi_band_features(train_amb, filterA, filterB, filterC);
    fire_train_feats = compute_multi_band_features(train_fire, filterA, filterB, filterC);

    %% === STEP 4: Set Threshold from Training Ratios ===
    amb_ratios = amb_train_feats(:,2) ./ (amb_train_feats(:,1) + amb_train_feats(:,3) + 1e-6);
    fire_ratios = fire_train_feats(:,2) ./ (fire_train_feats(:,1) + fire_train_feats(:,3) + 1e-6);
    threshold = (mean(amb_ratios) + mean(fire_ratios)) / 2;

    % Prepare KNN classifier on training data
    train_feats = [amb_train_feats; fire_train_feats];
    train_labels = [repmat({'ambulance'}, size(amb_train_feats, 1), 1); ...
                    repmat({'firetruck'}, size(fire_train_feats, 1), 1)];
    ratio_feats = train_feats(:,2) ./ (train_feats(:,1) + train_feats(:,3) + 1e-6);
    knn_model = fitcknn(ratio_feats, train_labels, 'NumNeighbors', 3);

    % Plot ratio distributions
    figure;
    histogram(amb_ratios, 20, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
    histogram(fire_ratios, 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
    title('Energy Ratio Distribution (Band B / others)');
    xlabel('Ratio'); ylabel('Count'); legend('Ambulance','Firetruck');

    %% === STEP 5: Test and Evaluate ===
    fprintf('Classifying test data...\n');
    amb_results = classify_test_multi(test_amb, filterA, filterB, filterC, threshold, knn_model, 'ambulance');
    fire_results = classify_test_multi(test_fire, filterA, filterB, filterC, threshold, knn_model, 'firetruck');

    all_results = [amb_results; fire_results];
    correct = sum(strcmp(all_results(:,2), all_results(:,3)) | strcmp(all_results(:,2), all_results(:,5)));
    accuracy = (correct / size(all_results, 1)) * 100;

    % Display results
    result_table = cell2table(all_results, ...
        'VariableNames', {'File', 'True_Label', 'Filter_Prediction', 'BandB_Ratio', 'KNN_Prediction'});
    disp(result_table);
    fprintf('\nFinal Combined Classification Accuracy: %.2f%%\n', accuracy);
end

%% === Supporting Function: Spectrum Average ===
function avg_spectrum = average_spectrum(folder, fs)
    files = dir(fullfile(folder, '*.wav'));
    N = 2048;
    total = zeros(N/2+1, 1);
    for i = 1:length(files)
        [y, ~] = audioread(fullfile(files(i).folder, files(i).name));
        if size(y,2) > 1, y = mean(y,2); end
        y = y(1:min(end, fs)); y = y .* hann(length(y));
        Y = abs(fft(y, N));
        total = total + Y(1:N/2+1);
    end
    avg_spectrum = total / length(files);
end

%% === Supporting Function: Filtered Energy Features ===
function feats = compute_multi_band_features(folder, fA, fB, fC)
    files = dir(fullfile(folder, '*.wav'));
    feats = zeros(length(files), 3);
    for i = 1:length(files)
        [y, ~] = audioread(fullfile(files(i).folder, files(i).name));
        if size(y,2) > 1, y = mean(y,2); end
        y = y(1:min(end, 44100)); y = y / max(abs(y));
        EA = sum((filter(fA, y)).^2);
        EB = sum((filter(fB, y)).^2);
        EC = sum((filter(fC, y)).^2);
        feats(i,:) = [EA, EB, EC];
    end
end

%% === Supporting Function: Classify Test Files ===
function results = classify_test_multi(folder, fA, fB, fC, threshold, knn_model, true_label)
    files = dir(fullfile(folder, '*.wav'));
    results = cell(length(files), 5);
    for i = 1:length(files)
        [y, ~] = audioread(fullfile(files(i).folder, files(i).name));
        if size(y,2) > 1, y = mean(y,2); end
        y = y(1:min(end, 44100)); y = y / max(abs(y));
        EA = sum((filter(fA, y)).^2);
        EB = sum((filter(fB, y)).^2);
        EC = sum((filter(fC, y)).^2);
        ratio = EB / (EA + EC + 1e-6);
        predicted = 'firetruck';
        if ratio > threshold
            predicted = 'ambulance';
        end
        knn_pred = predict(knn_model, ratio);

        results{i,1} = files(i).name;
        results{i,2} = true_label;
        results{i,3} = predicted;
        results{i,4} = ratio;
        results{i,5} = knn_pred;
    end
end


Fs     = 125;
ecgRaw = values(:);
% один раз
state = initAlgState(Fs);

chunkSize = Fs * 2;           % как будто каждые 2 секунды нам приходят 400 отсчётов

pos = 1;
while pos <= numel(ecgRaw)
    chunk = ecgRaw(pos : min(numel(ecgRaw), pos + chunkSize - 1));

    state = processEcgChunk(chunk, state);

    pos = pos + chunkSize;
end

% после цикла в state лежит всё:
allPeakIdx    = state.allPeakIdx;
normEcgGlobal = state.normEcgGlobal;
thrGlobal     = state.thrGlobal;
HR_time       = state.HR_time;
HR_values     = state.HR_values;

%% --- Фильтр для отображения сигнала врачам (0.5–40 Гц) ---
[bDisp, aDisp] = butter(5, [0.5 20]/(Fs/2), 'bandpass');  % порядок 4
ecgDisp = nan(N,1);
ziDisp = [];   % будет заполнено при первом вызове filter
chunkSize = VALUES_TO_CALCULATE_SHIFT;  % имитация прихода данных каждые 2 c
idxStart = 1;
while idxStart <= N
    idxEnd = min(N, idxStart + chunkSize - 1);

    [yChunk, ziDisp] = filter(bDisp, aDisp, ecgRaw(idxStart:idxEnd), ziDisp);
    ecgDisp(idxStart:idxEnd) = yChunk;

    idxStart = idxStart + chunkSize;
end
figure;
plot(t, ecgDisp);
grid on;
xlabel('Time, s');
ylabel('mV (filtered)');
title('ECG для отображения (0.5–20 Гц)');


figure;

% 1) Сырой сигнал с R-пиками
subplot(2,1,1);
plot(t, ecgRaw); hold on;
plot(t(allPeakIdx), ecgRaw(allPeakIdx), 'ro');
grid on;
xlabel('Time, s');
ylabel('Amplitude');
title('Raw ECG с R-пиками');
hold off;

% 2) Фильтрованный сигнал + порог + те же пики
subplot(2,1,2);
plot(t, normEcgGlobal); hold on;
plot(t(allPeakIdx), normEcgGlobal(allPeakIdx), 'ro');   % пики на фильтрованном
plot(t, thrGlobal, 'r--', 'LineWidth', 1);              % линия порога
grid on;
xlabel('Time, s');
ylabel('Filtered');
title('Filtered ECG (normEcgGlobal) + threshold');
hold off;

%% Расчёт ЧСС по накопленным пикам (моментальный ЧСС)
tPeaks = t(allPeakIdx);       % времена всех найденных R
RR     = diff(tPeaks);        % RR-интервалы (сек)
HR     = round(60 ./ RR);            % мгновенная ЧСС (bpm)

% Время для каждой точки ЧСС — середина RR-интервала
tHR = (tPeaks(1:end-1) + tPeaks(2:end)) / 2;

figure;
plot(tHR, HR, '-o');
grid on;
xlabel('Time, s');
ylabel('HR, bpm');
title('ЧСС из RR (моментальная ЧСС)');

% Отображаем нормально посчитанный ЧСС
figure;
plot(HR_time, HR_values, '-o');
grid on;
xlabel('Time, s');
ylabel('HR, bpm');
title('ЧСС (онлайн-расчёт при каждом R-пике)');




% Compare different ODE solvers with baseline
addpath(genpath("matlab2tikz/src/"))

%% general parameters
% This section is required to compute r and v from the data downloaded from
% NORAD. If you got the r0 and v0, you can just input them manually and
% delete this section

fprintf('loading...');
%% parameters for J2 and drag
whichconsts = 84;
Cd = 2.2;
mu = 398600.5;
Re = 6378.137;
J2 = 0.01082635854;

%% these values are from the CATCH paper
deltas = [8,16]; %number of sections per orbit
Ns = [8 16]; %number of points per section
tmaxs = 60*60*24*14; %14 days

tspan = [];
segment_duration = orbital_period / num_segments;
total_segments = ceil(total_time / segment_duration);

for segment = 1:total_segments
    t_start = (segment - 1) * segment_duration;  % MATLAB is 1-based, so segment-1
    t_end = min(t_start + segment_duration, total_time);  % Handle partial segment at end

    segment_points = gaussLobattoPoints(n_points, t_start, t_end);

    if segment == 1
        % First segment: include all points
        tspan = [tspan, segment_points];
    else
        % Subsequent segments: skip first point
        tspan = [tspan, segment_points(2:end)];
    end
end

path_to_csv = '../results/';
%% create a vector struct of all satellites
satsfilename = '../c++\ code/satellites.json';
satellites = readstruct(satsfilename);
fprintf("done\n")
for i = 1:length(satellites)
    satname = satellites(i).name;
    A = satellites(i).A*1e-6;
    m = satellites(i).m;
    r0 = satellites(i).r0;
    v0 = satellites(i).v0;
    for N = Ns
        for delta = deltas
            fprintf(satname+"_N"+N+"_delta"+delta+"...")

            %% Prepare time point list

            [a,e,inc,O,w,f0] = kep_elements(r0,v0,mu); %transfer the r and v to
            %  keplerian elements
            Sec = 2*pi*sqrt((a^3)/mu)/delta; %These is the orbital period of the
            % satellite, devided by delta. It's the time of one section

            %% Generate baseline
            fprintf("generating baseline...")

            options = odeset('RelTol',1e-14,'AbsTol', 1e-15,'MaxStep',Sec/(N*10));
            [~,rbaseline] = ode89(@(t,x)orbit_eq_J2_drag(t,x,mu,Cd,A,m,Re,J2),tspan,[r0 v0],options);
            
            fprintf("done!\n")
            %% load the relevant data from files:
            rk4res = readmatrix([path_to_csv_file,satname,'_RK4_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            rk8res = readmatrix([path_to_csv_file,satname,'_RK8_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            ode45res = readmatrix([path_to_csv_file,satname,'_ODE45_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            ode78res = readmatrix([path_to_csv_file,satname,'_ODE78_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            ode113res = readmatrix([path_to_csv_file,satname,'_ODE113_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            MPCIres = readmatrix([path_to_csv_file,satname,'_MPCI_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            
            err4 = rbaseline - rk4res(:,2:end);
            err8 = rbaseline - rk8res(:,2:end);
            err45 = rbaseline - ode45res(:,2:end);
            err78 = rbaseline - ode78res(:,2:end);
            err113 = rbaseline - ode113res(:,2:end);
            errMPCI = rbaseline - MPCIres(:,2:end);

            normr4 = sqrt(sum(err4(:,1:3).^2,2))';
            normr8 = sqrt(sum(err8(:,1:3).^2,2))';
            normr45 = sqrt(sum(err45(:,1:3).^2,2))';
            normr78 = sqrt(sum(err78(:,1:3).^2,2))';
            normr113 = sqrt(sum(err113(:,1:3).^2,2))';
            normrMPCI = sqrt(sum(errMPCI(:,1:3).^2,2))';

            plot_tikz_figure(tspan,[normr4;normr8;normr45;normr78;normr113;normrMPCI],"../results/"+satname+"_J2_delta"+delta+"_N"+N)

            

        end
    end
end

%% Play sound
fs = 8192;
toneduration = 0.1;
spaceduration = 0.05;
tonefreq = 800;
nbeeps = 10;

t = linspace(0,toneduration,round(toneduration*fs));
y=0.8*sin(2*pi*tonefreq*t);
ys = zeros(1,round(spaceduration*fs));

Y=[repmat([y ys],[1 nbeeps-1]) y];
sound(Y,fs);

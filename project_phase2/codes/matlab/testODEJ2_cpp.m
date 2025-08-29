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
mu = 398600.4418;
Re = 6378.137;
J2 = 1.08263e-3;

%% these values are from the CATCH paper
deltas = [8,16]; %number of sections per orbit
Ns = 16; %number of points per section
tmax = 60*60*24*30; %14 days

path_to_csv = '../c++ codes/results/';
%% create a vector struct of all satellites
satsfilename = '../c++ codes/satellites.json';
jsonStr = fileread(satsfilename);
satellites = jsondecode(jsonStr);
fprintf("done\n")
for i = 1:length(satellites)
%i=2;
    satname = satellites(i).name;
    A = satellites(i).A*1e-6;
    m = satellites(i).m;
    r0 = satellites(i).r0;
    v0 = satellites(i).v0;
    for N = Ns
        for delta = deltas
            fprintf(satname+"_N"+N+"_delta"+delta+"...")

            %% Prepare time point list

            a = 1/(2/norm(r0)-norm(v0)^2/mu);
            %  keplerian elements
            Sec = 2*pi*sqrt((a^3)/mu)/delta; %These is the orbital period of the
            % satellite, devided by delta. It's the time of one section
            tspan = [];
            
            total_segments = ceil(tmax / Sec);

            for segment = 1:total_segments
                t_start = (segment - 1) * Sec;  % MATLAB is 1-based, so segment-1
                t_end = min(t_start + Sec, tmax);  % Handle partial segment at end

                segment_points = 0.5*(t_end-t_start)*(-cos((0:(N-1))/(N-1)*pi)+1)+t_start;
                if segment == 1
                    % First segment: include all points
                    tspan = [tspan, segment_points];
                else
                    % Subsequent segments: skip first point
                    tspan = [tspan, segment_points(2:end)];
                end
            end

            %% Generate baseline
            fprintf("generating baseline...")

            options = odeset('RelTol',1e-14,'AbsTol', 1e-15,'MaxStep',Sec/(N*10));
            [ttest,rbaseline] = ode89(@(t,x)orbit_eq_J2_drag(t,x,mu,Cd,A,m,Re,J2),tspan,[r0' v0'],options);
%             optMPCI.AbsTol = 1e-12;
%             optMPCI.RelTol = 1e-9;
%             optMPCI.N = 16;
%             optMPCI.Sec = Sec;
%             [ttestMPCI,rtestMPCI] = odeMPCI(@(t,x)orbit_eq_J2_drag(t,x,mu,Cd,A,m,Re,J2),tspan,[r0' v0'],optMPCI);
            
            fprintf("done!\n")
            %% load the relevant data from files:
            rk4res = readmatrix([path_to_csv,satname,'_RK4_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            rk8res = readmatrix([path_to_csv,satname,'_RK8_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            ode45res = readmatrix([path_to_csv,satname,'_ODE45_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            ode78res = readmatrix([path_to_csv,satname,'_ODE78_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            ode113res = readmatrix([path_to_csv,satname,'_ODE113_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            MPCIres = readmatrix([path_to_csv,satname,'_MPCI_final_positions_',num2str(delta),'_',num2str(N),'.csv']);
            %% compute errors
            errt = abs(ttest - MPCIres(:,1));
            err4 = rbaseline(:,1:3) - rk4res(:,2:end);
            err8 = rbaseline(:,1:3) - rk8res(:,2:end);
            err45 = rbaseline(:,1:3) - ode45res(:,2:end);
            err78 = rbaseline(:,1:3) - ode78res(:,2:end);
            err113 = rbaseline(:,1:3) - ode113res(:,2:end);
            errMPCI = rbaseline(:,1:3) - MPCIres(:,2:end);
            
            %% norms
            normr4 = sqrt(sum(err4(:,1:3).^2,2))';
            normr8 = sqrt(sum(err8(:,1:3).^2,2))';
            normr45 = sqrt(sum(err45(:,1:3).^2,2))';
            normr78 = sqrt(sum(err78(:,1:3).^2,2))';
            normr113 = sqrt(sum(err113(:,1:3).^2,2))';
            normrMPCI = sqrt(sum(errMPCI(:,1:3).^2,2))';
            %%
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

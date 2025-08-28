function plot_tikz_figure (tspan,MAT,fname,debug)

if nargin ==3
    debug = false;
end
figure
loglog(tspan,MAT,'LineWidth', 1);
legend('hide');
xlabel("Time (s)");
ylabel("Error (km)");
%legend(legend_cell,'location','southeast')
set(findall(gcf,'type','axes'),'fontsize',12);
set(findall(gcf,'type','text'),'fontSize',12);
grid on
cleanfigure('targetResolution',300);
matlab2tikz('filename',[char(fname) '.tikz'],'addLabels',false,'checkForUpdates',false,...
    'showInfo', false,'showWarnings',false,...
    'noSize',true);
saveas(gcf,[char(fname) '.pdf']);
if ~debug
    close(gcf);
end
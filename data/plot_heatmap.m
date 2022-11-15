seqs = readmatrix('binary_sequences_all.csv');
labels = readmatrix('flash_data_all.csv','range',[2 2]);
frontalis = seqs(labels==1,:);
knulli = seqs(labels==0,:);
carolinus = seqs(labels==2,:);
pyralis = seqs(labels==3,:);
bw = seqs(labels==4,:);
%%
figure()
pyralis_heatmap = nansum(pyralis)./size(pyralis,1);
bw_heatmap = nansum(bw)./size(bw,1);
knulli_heatmap = nansum(knulli)./size(knulli,1);
carolinus_heatmap = nansum(carolinus)./size(carolinus,1);
frontalis_heatmap = nansum(frontalis)./size(frontalis,1);
bcolor([bw_heatmap;pyralis_heatmap;carolinus_heatmap;frontalis_heatmap;knulli_heatmap])
xlim([1 200])
colorbar
set(gca,'dataaspectratio',[1 .05 1])
% caxis([0 1])
% colorbar
set(gca,'fontsize',18)
set(gca,'xtick',30.5:30:300.5);
set(gca,'xticklabel',1:10);
xlabel('Time (s)')
set(gca,'ytick',1.5:1:5.5)
set(gca,'yticklabel',fliplr({'P. knulli','P. frontalis','P. carolinus','P. pyralis','B. wickershamorum' }))

%% plot individual pop refs
figure()
subplot(3,1,3)
knulli_heatmap = nansum(knulli)./size(knulli,1);
carolinus_heatmap = nansum(carolinus)./size(carolinus,1);
frontalis_heatmap = nansum(frontalis)./size(frontalis,1);
all_data = (vertcat(knulli_heatmap,carolinus_heatmap,frontalis_heatmap));
knulli_lit = horzcat(ones(1,round(.11*30)),zeros(1,round(.22*30)),ones(1,round(.11*30)),zeros(1,round(.376*30)),ones(1,round(.11*30)));
knulli_lit = horzcat(knulli_lit,zeros(1,100));
knulli_lit = knulli_lit(1:(3*30));
h2=patch('xdata',[1  round(.11*30) round(.11*30) 1],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5);
hold on
patch('xdata',[11  13 13 11],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[25 27 27 25],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)

xlim([1 200])
h1=plot(knulli_heatmap,'linewidth',2);
ylabel('Intensity')
legend([h1,h2],{'data','ref'})
set(gca,'dataaspectratio',[1 .05 1])
set(gca,'fontsize',18)
set(gca,'xtick',30.5:30:300.5);
set(gca,'xticklabel',1:10);
xlabel('Time (s)')
title('P. knulli')


subplot(3,1,2)
carolinus_lit = horzcat(ones(1,round(.182*30)),zeros(1,round(.42*30)));
carolinus_lit = horzcat(repmat(carolinus_lit,1,9));
h2=patch('xdata',[1  5 5 1],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5);
hold on
patch('xdata',[19  23 23 19],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[37 41 41 37],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[55 59 59 55],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[73 77 77 73],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[91 95 95 91],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[109 113 113 109],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[127 131 131 127],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[145 149 149 145],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
h1=plot(carolinus_heatmap,'linewidth',2);
ylabel('Intensity')
xlim([1 200])
legend([h1,h2],{'data','ref'})
set(gca,'dataaspectratio',[1 .05 1])
set(gca,'fontsize',18)
set(gca,'xtick',30.5:30:300.5);
set(gca,'xticklabel',1:10);
xlabel('Time (s)')
title('P. carolinus')


subplot(3,1,1)
frontalis_lit = horzcat(ones(1,4),zeros(1,14));
frontalis_lit = horzcat(repmat(frontalis_lit,1,20));
h2=patch('xdata',[1  4 4 1],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5);
hold on
patch('xdata',[19 22 22 19],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[37 40 40 37],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[55 58 58 55],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[73 76 76 73],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[91 94 94 91],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[109 112 112 109],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[127 130 130 127],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[145 148 148 145],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[163 166 166 163],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[181 184 184 181],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
patch('xdata',[199 202 202 199],'ydata',[0 0 1 1] ,'facecolor',[0.8500 0.3250 0.0980],'edgecolor','none','facealpha',0.5)
h1=plot(frontalis_heatmap,'linewidth',2);
ylabel('Intensity')
xlim([1 200])
legend([h1,h2],{'data','ref'})
set(gca,'dataaspectratio',[1 .05 1])
set(gca,'fontsize',18)
set(gca,'xtick',30.5:30:300.5);
set(gca,'xticklabel',1:10);
xlabel('Time (s)')
title('P. frontalis')


















function h = bcolor(inmat)
% provides a balanced color plot (no row/cols left out) with no edge lines
    if ~ismatrix(inmat)
        error('input matrix must be two-dimensional'); 
    end
    
    pad = mean(mean(inmat));
    h = pcolor(padarray(inmat,[1 1],pad,'post'));
    set(h, 'EdgeColor', 'none');
    
end
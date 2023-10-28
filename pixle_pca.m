function [x,pcaim]=pixle_pca(pca_len,flag, plotf)
% call the function as: [x,pcaim]=pixle_pca(0,0, 0)
f=uigetdir('C:\','Select the folder of images.');
folders=dir(f);
filenum=length(folders);

for j=1:filenum
    if ~folders(j).isdir
        imageloc=fullfile(folders(j).folder, folders(j).name);
        im=imread(imageloc);
        [r,c,d]=size(im);
        if plotf
            figure
            imshow(im(:,:,1))
            figure
            imshow(im(:,:,2))
            figure
            imshow(im(:,:,3))
            figure
        end
        x=[reshape(im(:,:,1),[numel(im(:,:,1)), 1]) reshape(im(:,:,2),[numel(im(:,:,2)), 1]) reshape(im(:,:,3),[numel(im(:,:,3)), 1])];
        x=double(x);
        xcorr=corr(x);
        if ~flag
            if d==3
                xmax=max(x);
                x=x./repmat(xmax,[numel(im(:,:,1)), 1]);
                [coeff,score,latent]=pca(x,'numComponents', 3);
                scoreim=(score-repmat(min(score), [numel(im(:,:,1)), 1])).*repmat(xmax,[numel(im(:,:,1)), 1]);
                x=score;
            end
        else
            win_length=floor(length(x(:,1))/pca_len);
            ymax=zeros(length(x(:,1)),3);
            scoreim=zeros(length(x(:,1)),3);
            for i=1:win_length
                y=x(pca_len*(i-1)+1:pca_len*i,:);

                ymax(pca_len*(i-1)+1:pca_len*i,:)=repmat(max(y),[pca_len 1]);
                y=y./ymax(pca_len*(i-1)+1:pca_len*i,:);

                [coeff,score,latent]=pca(y,'numComponents', 3);

                scoreim(pca_len*(i-1)+1:pca_len*i,:)=(score-repmat(min(score),[length(y(:,:,1)),1])).*ymax(pca_len*(i-1)+1:pca_len*i,:);%
                x(pca_len*(i-1)+1:pca_len*i,:)=score;
            end
            if pca_len*win_length<length(x(:,1))
                y=x(pca_len*win_length+1:end,:);
                [yr,yc]=size(y);
                ymax(pca_len*win_length+1:end,:)=repmat(max(y),[yr 1]);
                y=y./ymax(pca_len*win_length+1:end,:);
                [coeff,score,latent]=pca(y,'numComponents', 3);
                scoreim(pca_len*win_length+1:end,:)=(score-repmat(min(score),[length(y(:,:,1)),1])).*ymax(pca_len*win_length+1:end,:);
                x(pca_len*win_length+1:end,:)=score;
            end
        end
        score=[];
        scoreim=(scoreim-min(scoreim))./(max(scoreim))*255;
        scoreim=uint8(scoreim);

        corr(x);
        pcaim(:,:,1)=reshape(scoreim(:,1),[ r c]);
        pcaim(:,:,2)=reshape(scoreim(:,2),[ r c]);
        pcaim(:,:,3)=reshape(scoreim(:,3),[ r c]);
        if plotf
            imshow(pcaim(:,:,1))
            figure
            imshow(pcaim(:,:,2))
            figure
            imshow(pcaim(:,:,3))
        end
        imwrite(pcaim,fullfile(folders(j).folder, strcat('pca', folders(j).name)));
    end
end
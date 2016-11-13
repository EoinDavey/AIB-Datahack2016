function normalise(filename)

X_data = csvread(filename);

rows = size(X_data,1);
cols = size(X_data, 2);

disp(size(X_data))

mn = mean(X_data);
dev = std(X_data);
for i = 1:cols,
    for j = 1:rows,
        if(X_data(j,i)==0),
            X_data(j,i) = mn(i);
        end;
    end;
end;
csvwrite(filename,X_data);

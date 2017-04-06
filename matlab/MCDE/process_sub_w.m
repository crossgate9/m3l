function [] = process_sub_w(hash, drfun, origD, targD)

if ~ exist('cache', 'dir'),
    error(message('no cache folder found.'))
end

origin_folder = sprintf('cache/%s/%s_%d', hash, drfun, origD);
target_folder = sprintf('cache/%s/%s_%d', hash, drfun, targD);

if ~ exist(origin_folder, 'dir'),
    error(message('original folder not found'))
end

if ~ exist(target_folder, 'dir'),
    mkdir(target_folder);
end

query = sprintf('%s/*.mat', origin_folder);
files = dir(query);
origin_filename_template = strcat(origin_folder, '/%s');
target_filename_template = strcat(target_folder, '/%s');

for file = files',
    origin_filename = sprintf(origin_filename_template, file.name);
    target_filename = sprintf(target_filename_template, file.name);
    load(origin_filename);
    W = get_sub_w(W, targD);
    save(target_filename, 'W');
end
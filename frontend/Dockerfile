FROM nginx:alpine

COPY . /usr/share/nginx/html

EXPOSE 8009

# nginx defaults to listening on 80 — repoint it to 8009 so the container
# matches the port it's actually published on.
RUN sed -i 's/listen\s*80;/listen 8009;/' /etc/nginx/conf.d/default.conf

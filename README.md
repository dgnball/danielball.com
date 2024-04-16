```bash
brew install rbenv
echo 'eval "$(rbenv init -)"' >> ~/.zshrc
source ~/.zshrc
rbenv install -l
rbenv install 3.3.0
rbenv global 3.3.0
source ~/.zshrc
ruby -v
gem install jekyll bundler
```

To preview your Jekyll site locally, you'll need to serve it using Jekyll's built-in server, which will allow you to view the site in your web browser as if it were live online. Here’s how you can do it:

Open Terminal or Command Prompt: Navigate to your project directory where your Jekyll site is located. You can do this using the cd (change directory) command followed by the path to your project directory. For example:

```bash
cd jekyll-site
bundle exec jekyll serve --livereload
```
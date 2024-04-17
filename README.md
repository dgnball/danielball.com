General instructions to set up Ruby and Jekyll on macOS:

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

To run locally, open Terminal or Command Prompt in the root of this repo and run the following:

```bash
bundle exec jekyll serve --livereload
```